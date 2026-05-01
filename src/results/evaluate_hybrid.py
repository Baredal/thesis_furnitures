import csv
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "src" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR / "src"))
from ml.model import SiameseResnet18, SiameseEfficientNetB3
from retrieval.histogram_filter import compute as compute_hist

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH    = 64
MARGIN   = 1

EMBED_W = 0.5
HIST_W  = 0.5

RECALL_K       = [1, 2, 3, 4, 5]
SCENE_RECALL_K = [5, 10, 15, 20, 25, 30]
PR_K           = [1, 3, 5, 10, 15, 20, 30, 50]

ROOMS = ["bedrooms", "living_rooms"]

CHECKPOINTS = {
    "bedrooms":     "v3/best_model_v3_resnet18_new_data.pt",
    "living_rooms": "v3/best_model_v3_resnet18_liv_rooms.pt",
}

COLORS = {
    "ResNet18 backbone (ImageNet)":         "#4c72b0",
    "EfficientNet-B3 backbone (ImageNet)":  "#c44e52",
    "ResNet18 fine-tuned":                  "#55a868",
}
LS = {
    "ResNet18 backbone (ImageNet)":         "--",
    "EfficientNet-B3 backbone (ImageNet)":  ":",
    "ResNet18 fine-tuned":                  "-",
}

_ZERO_HIST = np.zeros(96, dtype=np.float32)


def _pretty_cat(cat: str) -> str:
    if cat in ("small_storage", "large_storage"):
        return "storage"
    return cat


val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class TripletDataset(Dataset):
    def __init__(self, csv_path: Path, image_root: Path):
        self.image_root = image_root
        self.triplets = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.triplets.append((
                    (row["anchor_id"],   row["anchor_category"]),
                    (row["positive_id"], row["positive_category"]),
                    (row["negative_id"], row["negative_category"]),
                ))
        size = (IMG_SIZE, IMG_SIZE)
        unique_paths = {
            image_root / cat / f"{fid}.jpg"
            for triplet in self.triplets
            for fid, cat in triplet
        }
        print(f"  Pre-loading {len(unique_paths)} images...", end=" ", flush=True)
        self._cache: dict[Path, Image.Image] = {}
        for p in unique_paths:
            try:
                self._cache[p] = Image.open(p).convert("RGB").resize(size, Image.BILINEAR)
            except Exception:
                pass
        print("done.")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        (a_id, a_cat), (p_id, p_cat), (n_id, n_cat) = self.triplets[idx]
        def load(fid, cat):
            p = self.image_root / cat / f"{fid}.jpg"
            img = self._cache.get(p)
            if img is None:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            return val_tf(img)
        return load(a_id, a_cat), load(p_id, p_cat), load(n_id, n_cat)


def load_golden_gallery(triplet_dir: Path, image_dir: Path):
    with open(triplet_dir / "scene_split.json", encoding="utf-8") as f:
        golden_scenes = set(json.load(f)["golden_scenes"])
    with open(image_dir / "general_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)["furnitures"]
    item_list      = [it for it in manifest if it["scene"] in golden_scenes]
    item_to_scene  = {it["furniture_id"]: it["scene"] for it in item_list}
    scene_to_items = defaultdict(set)
    for it in item_list:
        scene_to_items[it["scene"]].add(it["furniture_id"])
    return item_list, item_to_scene, scene_to_items


def build_gallery_hists(item_list, image_dir: Path) -> dict:
    hists, failed = {}, 0
    for it in item_list:
        fid = it["furniture_id"]
        img_path = image_dir / it["category"] / it["image_name"]
        try:
            hists[fid] = compute_hist(img_path)
        except Exception:
            failed += 1
    print(f"  Gallery hists: {len(hists)} ok, {failed} failed")
    return hists


def build_triplet_hists(csv_path: Path, image_dir: Path) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    hists, failed = {}, 0
    cols = [("anchor_id", "anchor_category"),
            ("positive_id", "positive_category"),
            ("negative_id", "negative_category")]
    for row in rows:
        for id_col, cat_col in cols:
            fid = row[id_col]
            if fid not in hists:
                img_path = image_dir / row[cat_col] / f"{fid}.jpg"
                try:
                    hists[fid] = compute_hist(img_path)
                except Exception:
                    failed += 1
    print(f"  Triplet hists: {len(hists)} ok, {failed} failed")
    return hists


def hybrid_score(
    anc_emb:   torch.Tensor,
    anc_hist:  np.ndarray,
    gal_embs:  torch.Tensor,
    gal_hists: np.ndarray,
) -> np.ndarray:
    cos = (gal_embs @ anc_emb).numpy().astype(np.float32)
    bc  = (gal_hists @ anc_hist).astype(np.float32)
    return EMBED_W * cos + HIST_W * bc


class BackboneOnlyWrapper(nn.Module):
    """L2-normalised backbone features, bypassing the randomly-init embedding head."""
    def __init__(self, siamese_model: nn.Module):
        super().__init__()
        self.backbone = siamese_model.backbone
        self.pool     = getattr(siamese_model, "pool", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if self.pool is not None:
            x = self.pool(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2, dim=1)


def build_models_cfg(room: str) -> dict:
    model_dir = BASE_DIR / "data" / "ml_data" / room / "models"
    return {
        "ResNet18 backbone (ImageNet)": {
            "cls": SiameseResnet18, "checkpoint": None,
            "emb_dim": 128, "backbone_only": True,
        },
        "EfficientNet-B3 backbone (ImageNet)": {
            "cls": SiameseEfficientNetB3, "checkpoint": None,
            "emb_dim": 128, "backbone_only": True,
        },
        "ResNet18 fine-tuned": {
            "cls": SiameseResnet18,
            "checkpoint": model_dir / CHECKPOINTS[room],
            "emb_dim": None, "backbone_only": False,
        },
    }


def load_model(cfg: dict) -> nn.Module:
    ckpt_path = cfg["checkpoint"]
    if ckpt_path is None:
        base = cfg["cls"](embedding_dim=cfg["emb_dim"], pretrained=True)
        if cfg.get("backbone_only", False):
            model = BackboneOnlyWrapper(base)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
                feat_dim = model(dummy).shape[1]
            print(f"  ImageNet backbone-only (no head), feature_dim={feat_dim}")
        else:
            model = base
            print(f"  ImageNet pretrained with embedding head, emb_dim={cfg['emb_dim']}")
    else:
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        c     = ckpt["config"]
        model = cfg["cls"](embedding_dim=c["embedding_dim"], pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Checkpoint epoch={ckpt['epoch']}, "
              f"val_loss={ckpt.get('val_loss','?'):.4f}, emb_dim={c['embedding_dim']}")
    return model.to(DEVICE).eval()


criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)


@torch.no_grad()
def embed_triplets(model, dataloader):
    a_list, p_list, n_list = [], [], []
    total_loss, n = 0.0, 0
    for anchor, positive, negative in dataloader:
        imgs = torch.cat([anchor, positive, negative]).to(DEVICE)
        embs = model(imgs)
        a, p, ne = embs.chunk(3)
        a_list.append(a.cpu()); p_list.append(p.cpu()); n_list.append(ne.cpu())
        total_loss += criterion(a, p, ne).item() * anchor.size(0)
        n += anchor.size(0)
    a_embs = torch.cat(a_list)
    p_embs = torch.cat(p_list)
    n_embs = torch.cat(n_list)
    pos_d  = F.pairwise_distance(a_embs, p_embs).numpy()
    neg_d  = F.pairwise_distance(a_embs, n_embs).numpy()
    return a_embs, p_embs, n_embs, pos_d, neg_d, total_loss / max(n, 1)


def triplet_metrics_only(pos_d, neg_d, loss):
    correct = (neg_d > pos_d).sum()
    return {
        "loss":             loss,
        "triplet_accuracy": float(correct / len(pos_d)),
        "margin_satisfied": float(((neg_d - pos_d) > MARGIN).sum() / len(pos_d)),
        "pos_dist_mean":    float(pos_d.mean()),
        "neg_dist_mean":    float(neg_d.mean()),
        "dist_gap_mean":    float((neg_d - pos_d).mean()),
    }


def hybrid_recall_at_k(a_embs, p_embs, n_embs, csv_path, hists, k_values):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pair_groups = defaultdict(list)
    for i, row in enumerate(rows):
        pair_groups[(row["anchor_id"], row["positive_id"])].append(i)
    ranks = []
    for (a_id, p_id), indices in pair_groups.items():
        a_emb  = a_embs[indices[0]]
        a_hist = hists.get(a_id, _ZERO_HIST)
        p_emb  = p_embs[indices[0]]
        p_hist = hists.get(p_id, _ZERO_HIST)
        score_pos = (EMBED_W * float(a_emb @ p_emb)
                     + HIST_W * float(np.dot(a_hist, p_hist)))
        n_scores = []
        for i in indices:
            n_id   = rows[i]["negative_id"]
            n_emb  = n_embs[i]
            n_hist = hists.get(n_id, _ZERO_HIST)
            n_scores.append(EMBED_W * float(a_emb @ n_emb)
                            + HIST_W * float(np.dot(a_hist, n_hist)))
        ranks.append(1 + sum(1 for s in n_scores if s > score_pos))
    ranks = np.array(ranks)
    return {f"hybrid_triplet_recall@{k}": float((ranks <= k).mean()) for k in k_values}


def hybrid_mrr(a_embs, p_embs, n_embs, csv_path, hists):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pair_groups = defaultdict(list)
    for i, row in enumerate(rows):
        pair_groups[(row["anchor_id"], row["positive_id"])].append(i)
    rr = []
    for (a_id, p_id), indices in pair_groups.items():
        a_emb  = a_embs[indices[0]]
        a_hist = hists.get(a_id, _ZERO_HIST)
        p_emb  = p_embs[indices[0]]
        p_hist = hists.get(p_id, _ZERO_HIST)
        score_pos = (EMBED_W * float(a_emb @ p_emb)
                     + HIST_W * float(np.dot(a_hist, p_hist)))
        n_scores = []
        for i in indices:
            n_id   = rows[i]["negative_id"]
            n_emb  = n_embs[i]
            n_hist = hists.get(n_id, _ZERO_HIST)
            n_scores.append(EMBED_W * float(a_emb @ n_emb)
                            + HIST_W * float(np.dot(a_hist, n_hist)))
        rank = 1 + sum(1 for s in n_scores if s > score_pos)
        rr.append(1.0 / rank)
    return {"hybrid_MRR": float(np.mean(rr))}


def hybrid_scene_recall_at_k(a_embs, p_embs, n_embs, csv_path, hists, k_values):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    item_to_emb, item_to_hist = {}, {}
    scene_to_items_ = defaultdict(set)
    item_to_scene_  = {}
    for i, row in enumerate(rows):
        for id_col, sc_col, emb_t in [
            ("anchor_id",   "anchor_scene",   a_embs),
            ("positive_id", "positive_scene", p_embs),
            ("negative_id", "negative_scene", n_embs),
        ]:
            fid = row[id_col]
            if fid not in item_to_emb:
                item_to_emb[fid]  = emb_t[i]
                item_to_hist[fid] = hists.get(fid, _ZERO_HIST)
            scene_to_items_[row[sc_col]].add(fid)
            item_to_scene_[fid] = row[sc_col]
    all_ids   = list(item_to_emb.keys())
    gal_embs  = torch.stack([item_to_emb[fid] for fid in all_ids])
    gal_hists = np.stack([item_to_hist[fid] for fid in all_ids])
    hits = {k: [] for k in k_values}
    seen = set()
    for row in rows:
        a_id = row["anchor_id"]
        if a_id in seen:
            continue
        seen.add(a_id)
        mates = scene_to_items_[item_to_scene_[a_id]] - {a_id}
        if not mates:
            continue
        a_emb  = item_to_emb[a_id]
        a_hist = item_to_hist[a_id]
        a_idx  = all_ids.index(a_id)
        scores = hybrid_score(a_emb, a_hist, gal_embs, gal_hists)
        scores[a_idx] = -1.0
        top_order = np.argsort(-scores)
        for k in k_values:
            topk = {all_ids[j] for j in top_order[:k]}
            hits[k].append(float(bool(topk & mates)))
    return {f"hybrid_scene_recall@{k}": float(np.mean(v)) if v else 0.0
            for k, v in hits.items()}


@torch.no_grad()
def embed_gallery(model, item_list, image_dir: Path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    item_to_emb = {}
    for it in item_list:
        fid = it["furniture_id"]
        img_path = image_dir / it["category"] / it["image_name"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        emb = model(transform(img).unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
        item_to_emb[fid] = emb
    return item_to_emb


def hybrid_precision_recall(item_to_emb, item_to_scene, scene_to_items,
                            gallery_hists_dict, k_values):
    all_ids   = list(item_to_emb.keys())
    gal_embs  = torch.stack([item_to_emb[fid] for fid in all_ids])
    gal_hists = np.stack([gallery_hists_dict.get(fid, _ZERO_HIST) for fid in all_ids])
    prec = {k: [] for k in k_values}
    rec  = {k: [] for k in k_values}
    id_to_idx = {fid: i for i, fid in enumerate(all_ids)}
    for fid in all_ids:
        scene = item_to_scene.get(fid)
        if scene is None:
            continue
        gt = (scene_to_items[scene] - {fid}) & set(all_ids)
        if not gt:
            continue
        idx    = id_to_idx[fid]
        a_emb  = item_to_emb[fid]
        a_hist = gallery_hists_dict.get(fid, _ZERO_HIST)
        scores = hybrid_score(a_emb, a_hist, gal_embs, gal_hists)
        scores[idx] = -1.0
        order = np.argsort(-scores)
        for k in k_values:
            topk = {all_ids[j] for j in order[:k]}
            hits = len(topk & gt)
            prec[k].append(hits / k)
            rec[k].append(hits / len(gt))
    return (
        {f"hybrid_compat_precision@{k}": float(np.mean(v)) for k, v in prec.items()},
        {f"hybrid_compat_recall@{k}":    float(np.mean(v)) for k, v in rec.items()},
    )


def plot_hybrid_scene_recall(all_results, model_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in model_names:
        vals = [all_results[name].get(f"hybrid_scene_recall@{k}", 0) for k in SCENE_RECALL_K]
        ax.plot(SCENE_RECALL_K, vals, marker="o",
                label=name, color=COLORS[name], ls=LS[name], lw=2)
    ax.set_xlabel("K")
    ax.set_ylabel("Scene Recall@K")
    ax.set_title("Hybrid Scene Recall@K")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hybrid_openset_pr(all_results, model_names, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name in model_names:
        p_vals = [all_results[name].get(f"hybrid_compat_precision@{k}", 0) for k in PR_K]
        r_vals = [all_results[name].get(f"hybrid_compat_recall@{k}",    0) for k in PR_K]
        ax1.plot(PR_K, p_vals, marker="o",
                 label=name, color=COLORS[name], ls=LS[name], lw=2)
        ax2.plot(PR_K, r_vals, marker="o",
                 label=name, color=COLORS[name], ls=LS[name], lw=2)
    ax1.set_xlabel("K"); ax1.set_ylabel("Precision")
    ax1.set_title("Hybrid open-set Precision@K")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("K"); ax2.set_ylabel("Recall")
    ax2.set_title("Hybrid open-set Recall@K")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_for_room(room: str):
    print("\n" + "=" * 70)
    print(f"  ROOM: {room.upper()}   [hybrid eval]")
    print("=" * 70)

    triplet_dir = BASE_DIR / "data" / "ml_data" / room / "triplets_v3"
    image_dir   = BASE_DIR / "data" / "total" / room
    golden_csv  = triplet_dir / "golden_triplets.csv"

    if not golden_csv.exists():
        print(f"  [{room}] {golden_csv} not found, skipping.")
        return
    if not (BASE_DIR / "data" / "ml_data" / room / "models" / CHECKPOINTS[room]).exists():
        print(f"  [{room}] Checkpoint not found, skipping.")
        return

    print("Loading golden triplet dataset...")
    ds = TripletDataset(golden_csv, image_dir)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"  {len(ds):,} triplets ({len(dl)} batches)")

    print("Loading golden gallery...")
    gallery_items, item_to_scene, scene_to_items = load_golden_gallery(triplet_dir, image_dir)
    print(f"  {len(gallery_items)} items across {len(scene_to_items)} scenes")

    print("Pre-building histogram caches...")
    gallery_hists = build_gallery_hists(gallery_items, image_dir)
    triplet_hists = build_triplet_hists(golden_csv, image_dir)

    models_cfg  = build_models_cfg(room)
    all_results = {}

    for model_name, cfg in models_cfg.items():
        print(f"\n  {'-'*56}")
        print(f"  {model_name}")
        print(f"  {'-'*56}")
        model = load_model(cfg)

        print("  [Triplet-based] embedding-only distances...")
        a_embs, p_embs, n_embs, pos_d, neg_d, loss = embed_triplets(model, dl)
        metrics = {}
        metrics.update(triplet_metrics_only(pos_d, neg_d, loss))

        print("  [Triplet-based] Hybrid Recall@K + MRR...")
        metrics.update(hybrid_recall_at_k(a_embs, p_embs, n_embs, golden_csv,
                                          triplet_hists, RECALL_K))
        metrics.update(hybrid_mrr(a_embs, p_embs, n_embs, golden_csv, triplet_hists))

        print("  [Triplet-based] Hybrid Scene Recall@K...")
        metrics.update(hybrid_scene_recall_at_k(a_embs, p_embs, n_embs, golden_csv,
                                                triplet_hists, SCENE_RECALL_K))

        del a_embs, p_embs, n_embs
        gc.collect()

        print("  [Open-set] Hybrid P@K / R@K on gallery...")
        item_to_emb = embed_gallery(model, gallery_items, image_dir)
        print(f"  Embedded {len(item_to_emb)} gallery items")
        p_dict, r_dict = hybrid_precision_recall(
            item_to_emb, item_to_scene, scene_to_items, gallery_hists, PR_K,
        )
        metrics.update(p_dict)
        metrics.update(r_dict)

        del item_to_emb, model
        gc.collect()
        torch.cuda.empty_cache()

        all_results[model_name] = metrics
        print("  Done.")

    model_names = list(all_results.keys())

    print("\n\n" + "=" * 70)
    print(f"  HYBRID RESULTS — {room.upper()} GOLDEN SET")
    print("=" * 70)

    def _print(title, keys):
        col_w  = max(len(k) for k in keys) + 2
        name_w = max(len(n) for n in model_names) + 2
        header = f"{'Metric':<{col_w}}" + "".join(f"{n:>{name_w}}" for n in model_names)
        print(f"\n-- {title} " + "-" * max(0, len(header) - len(title) - 3))
        print(header)
        print("-" * len(header))
        for key in keys:
            vals = [all_results[n].get(key, float("nan")) for n in model_names]
            print(f"{key:<{col_w}}" + "".join(f"{v:>{name_w}.4f}" for v in vals))

    _print("Triplet metrics (embedding-only)",
           ["loss", "triplet_accuracy", "margin_satisfied",
            "pos_dist_mean", "neg_dist_mean", "dist_gap_mean"])
    _print("Hybrid Recall@K (rank positive among 5 hard negatives)",
           [f"hybrid_triplet_recall@{k}" for k in RECALL_K])
    _print("Hybrid MRR", ["hybrid_MRR"])
    _print("Hybrid Scene Recall@K (full gallery)",
           [f"hybrid_scene_recall@{k}" for k in SCENE_RECALL_K])
    _print("Hybrid open-set Precision@K (scene co-occurrence)",
           [f"hybrid_compat_precision@{k}" for k in PR_K])
    _print("Hybrid open-set Recall@K (scene co-occurrence)",
           [f"hybrid_compat_recall@{k}" for k in PR_K])

    json_path = OUTPUT_DIR / f"{room}_results_hybrid.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[{room}] Hybrid results saved → {json_path}")

    plot_hybrid_scene_recall(all_results, model_names,
                             OUTPUT_DIR / f"{room}_hybrid_scene_recall.png")
    plot_hybrid_openset_pr(all_results, model_names,
                           OUTPUT_DIR / f"{room}_hybrid_openset_pr.png")
    print(f"[{room}] Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Device : {DEVICE}")
    print(f"Weights: emb={EMBED_W}  hist={HIST_W}")
    for room in ROOMS:
        run_for_room(room)
    print("\nAll rooms done.")
