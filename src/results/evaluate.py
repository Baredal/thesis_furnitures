"""Evaluate triplet and retrieval metrics on the golden split for both rooms.

Models compared: ResNet18 (ImageNet baseline), EfficientNet-B3 (ImageNet baseline),
ResNet18 fine-tuned.

Usage:
    python src/results/evaluate.py
"""

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

IMG_SIZE = 224
BATCH    = 64
MARGIN   = 1

RECALL_K       = [1, 2, 3, 4, 5]
SCENE_RECALL_K = [5, 10, 15, 20, 25, 30]
PR_K           = [1, 3, 5, 10, 15, 20, 30, 50]

ROOMS = ["bedrooms", "living_rooms"]

CHECKPOINTS = {
    "bedrooms":     "v3/best_model_v3_resnet18_new_data.pt",
    "living_rooms": "v3/best_model_v3_resnet18_liv_rooms.pt",
}

COLORS = ["#4c72b0", "#c44e52", "#55a868"]


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
            p   = self.image_root / cat / f"{fid}.jpg"
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


def build_models_cfg(room: str) -> dict:
    model_dir = BASE_DIR / "data" / "ml_data" / room / "models"
    return {
        "ResNet18 pretrained": {
            "cls": SiameseResnet18, "checkpoint": None, "emb_dim": 128,
        },
        "EfficientNet-B3 pretrained": {
            "cls": SiameseEfficientNetB3, "checkpoint": None, "emb_dim": 128,
        },
        "ResNet18 fine-tuned": {
            "cls":        SiameseResnet18,
            "checkpoint": model_dir / CHECKPOINTS[room],
            "emb_dim":    None,
        },
    }


def load_model(cfg: dict) -> nn.Module:
    ckpt_path = cfg["checkpoint"]
    if ckpt_path is None:
        model = cfg["cls"](embedding_dim=cfg["emb_dim"], pretrained=True)
    else:
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        c     = ckpt["config"]
        model = cfg["cls"](embedding_dim=c["embedding_dim"], pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"    Loaded epoch {ckpt['epoch']}, val_loss={ckpt.get('val_loss', '?'):.4f}")
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


def triplet_metrics(pos_d, neg_d):
    correct = (neg_d > pos_d).sum()
    return {
        "triplet_accuracy": float(correct / len(pos_d)),
        "margin_satisfied": float(((neg_d - pos_d) > MARGIN).sum() / len(pos_d)),
        "pos_dist_mean":    float(pos_d.mean()),
        "neg_dist_mean":    float(neg_d.mean()),
        "dist_gap_mean":    float((neg_d - pos_d).mean()),
    }


def recall_at_k_metric(a_embs, p_embs, n_embs, k_values, csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pair_groups = defaultdict(list)
    for i, row in enumerate(rows):
        pair_groups[(row["anchor_id"], row["positive_id"])].append(i)
    ranks = []
    for indices in pair_groups.values():
        a = a_embs[indices[0]]
        p = p_embs[indices[0]]
        d_pos = F.pairwise_distance(a.unsqueeze(0), p.unsqueeze(0)).item()
        neg_dists = [F.pairwise_distance(a_embs[i].unsqueeze(0), n_embs[i].unsqueeze(0)).item()
                     for i in indices]
        ranks.append(1 + sum(1 for d in neg_dists if d < d_pos))
    ranks = np.array(ranks)
    return {f"triplet_recall@{k}": float((ranks <= k).mean()) for k in k_values}


def mrr_metric(a_embs, p_embs, n_embs, csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pair_groups = defaultdict(list)
    for i, row in enumerate(rows):
        pair_groups[(row["anchor_id"], row["positive_id"])].append(i)
    rr = []
    for indices in pair_groups.values():
        a = a_embs[indices[0]]
        p = p_embs[indices[0]]
        d_pos = F.pairwise_distance(a.unsqueeze(0), p.unsqueeze(0)).item()
        neg_dists = [F.pairwise_distance(a_embs[i].unsqueeze(0), n_embs[i].unsqueeze(0)).item()
                     for i in indices]
        rank = 1 + sum(1 for d in neg_dists if d < d_pos)
        rr.append(1.0 / rank)
    return {"MRR": float(np.mean(rr))}


def scene_recall_at_k_metric(a_embs, p_embs, n_embs, k_values, csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    item_to_emb     = {}
    scene_to_items_ = defaultdict(set)
    item_to_scene_  = {}
    for i, row in enumerate(rows):
        for id_col, scene_col, emb_t in [
            ("anchor_id",   "anchor_scene",   a_embs),
            ("positive_id", "positive_scene", p_embs),
            ("negative_id", "negative_scene", n_embs),
        ]:
            fid   = row[id_col]
            scene = row[scene_col]
            if fid not in item_to_emb:
                item_to_emb[fid] = emb_t[i]
            scene_to_items_[scene].add(fid)
            item_to_scene_[fid] = scene
    all_ids = list(item_to_emb.keys())
    gallery = torch.stack([item_to_emb[fid] for fid in all_ids])
    max_k   = max(k_values)
    hits    = {k: [] for k in k_values}
    seen    = set()
    for row in rows:
        a_id = row["anchor_id"]
        if a_id in seen:
            continue
        seen.add(a_id)
        mates = scene_to_items_[item_to_scene_[a_id]] - {a_id}
        if not mates:
            continue
        a_idx = all_ids.index(a_id)
        dists = torch.cdist(gallery[a_idx].unsqueeze(0), gallery, p=2).squeeze(0)
        dists[a_idx] = float("inf")
        for k in k_values:
            topk_k = {all_ids[i] for i in dists.topk(k, largest=False).indices.tolist()}
            hits[k].append(float(bool(topk_k & mates)))
        del dists
    return {f"scene_recall@{k}": float(np.mean(v)) if v else 0.0 for k, v in hits.items()}


@torch.no_grad()
def embed_gallery(model, item_list, image_dir: Path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    item_to_emb = {}
    for it in item_list:
        fid      = it["furniture_id"]
        img_path = image_dir / it["category"] / it["image_name"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        emb    = model(tensor).squeeze(0).cpu()
        item_to_emb[fid] = emb
    return item_to_emb


def precision_recall_at_k(item_to_emb, item_to_scene, scene_to_items, k_values):
    all_ids = list(item_to_emb.keys())
    gallery = torch.stack([item_to_emb[fid] for fid in all_ids])
    id_to_idx = {fid: i for i, fid in enumerate(all_ids)}
    prec = {k: [] for k in k_values}
    rec  = {k: [] for k in k_values}
    for fid in all_ids:
        scene = item_to_scene.get(fid)
        if scene is None:
            continue
        gt = (scene_to_items[scene] - {fid}) & set(all_ids)
        if not gt:
            continue
        idx   = id_to_idx[fid]
        dists = torch.cdist(gallery[idx].unsqueeze(0), gallery, p=2).squeeze(0)
        dists[idx] = float("inf")
        sorted_ids = [all_ids[i] for i in dists.argsort().tolist()]
        del dists
        for k in k_values:
            top_k_set = set(sorted_ids[:k])
            hits = len(top_k_set & gt)
            prec[k].append(hits / k)
            rec[k].append(hits / len(gt))
    return (
        {f"compat_precision@{k}": float(np.mean(v)) for k, v in prec.items()},
        {f"compat_recall@{k}":    float(np.mean(v)) for k, v in rec.items()},
    )


def plot_distance_distributions(pos_neg_dists, out_path: Path):
    n_models = len(pos_neg_dists)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]
    for ax, (name, (pos_d, neg_d)) in zip(axes, pos_neg_dists.items()):
        ax.hist(pos_d, bins=60, alpha=0.6,
                label=f"Positive (mean={pos_d.mean():.3f})", color="#2196F3")
        ax.hist(neg_d, bins=60, alpha=0.6,
                label=f"Negative (mean={neg_d.mean():.3f})", color="#F44336")
        ax.axvline(MARGIN, color="k", ls="--", lw=1, label=f"Margin={MARGIN}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("L2 distance")
        ax.legend()
    axes[0].set_ylabel("Count")
    fig.suptitle("Positive vs Negative distance distributions — test set", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scene_recall(all_results, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, metrics), color in zip(all_results.items(), COLORS):
        vals = [metrics[f"scene_recall@{k}"] for k in SCENE_RECALL_K]
        ax.plot(SCENE_RECALL_K, vals, marker="o", label=name, color=color)
    ax.set_xlabel("K"); ax.set_ylabel("Scene Recall@K")
    ax.set_title("Scene Recall@K — full gallery")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_precision_recall(pr_curves, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for (name, (prec, rec)), color in zip(pr_curves.items(), COLORS):
        k_vals = sorted(int(k.split("@")[1]) for k in prec)
        ax1.plot(k_vals, [prec[f"compat_precision@{k}"] for k in k_vals],
                 marker="o", label=name, color=color)
        ax2.plot(k_vals, [rec[f"compat_recall@{k}"] for k in k_vals],
                 marker="o", label=name, color=color)
    ax1.set_xlabel("K"); ax1.set_ylabel("Precision@K")
    ax1.set_title("Precision@K — scene co-occurrence GT")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("K"); ax2.set_ylabel("Recall@K")
    ax2.set_title("Recall@K — scene co-occurrence GT")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_key_metrics(all_results, out_path: Path):
    model_names = list(all_results.keys())
    key_metrics = [
        "triplet_accuracy", "MRR", "triplet_recall@1", "triplet_recall@3",
        "scene_recall@10", "compat_precision@10", "compat_recall@10",
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(key_metrics))
    w = 0.25
    for i, (name, color) in enumerate(zip(model_names, COLORS)):
        vals = [all_results[name].get(m, 0) for m in key_metrics]
        bars = ax.bar(x + i * w, vals, w, label=name, color=color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x + w)
    ax.set_xticklabels(key_metrics, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Key metrics comparison — golden (test) set")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_section(title, keys, all_results, model_names):
    col_w = max(len(k) for k in keys) + 2
    name_w = max(len(n) for n in model_names) + 2
    header = f"{'Metric':<{col_w}}" + "".join(f"{n:>{name_w}}" for n in model_names)
    print(f"\n-- {title} " + "-" * max(0, len(header) - len(title) - 3))
    print(header)
    print("-" * len(header))
    for key in keys:
        vals = [all_results[n].get(key, float("nan")) for n in model_names]
        row  = f"{key:<{col_w}}" + "".join(f"{v:>{name_w}.4f}" for v in vals)
        print(row)


def run_for_room(room: str):
    print("\n" + "=" * 70)
    print(f"  ROOM: {room.upper()}")
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

    models_cfg    = build_models_cfg(room)
    all_results   = {}
    pr_curves     = {}
    pos_neg_dists = {}

    for model_name, cfg in models_cfg.items():
        print(f"\n  {'-'*56}")
        print(f"  {model_name}")
        print(f"  {'-'*56}")
        model = load_model(cfg)

        print("  Running triplet evaluation...")
        a_embs, p_embs, n_embs, pos_d, neg_d, loss = embed_triplets(model, dl)
        pos_neg_dists[model_name] = (pos_d.copy(), neg_d.copy())

        metrics = {"loss_raw": loss}
        metrics.update(triplet_metrics(pos_d, neg_d))
        metrics.update(recall_at_k_metric(a_embs, p_embs, n_embs, RECALL_K, golden_csv))
        metrics.update(mrr_metric(a_embs, p_embs, n_embs, golden_csv))
        metrics.update(scene_recall_at_k_metric(a_embs, p_embs, n_embs, SCENE_RECALL_K, golden_csv))

        del a_embs, p_embs, n_embs
        gc.collect()

        print("  Embedding gallery for P@K / R@K...")
        item_to_emb = embed_gallery(model, gallery_items, image_dir)
        print(f"  Embedded {len(item_to_emb)} gallery items")
        p_dict, r_dict = precision_recall_at_k(item_to_emb, item_to_scene, scene_to_items, PR_K)
        metrics.update(p_dict)
        metrics.update(r_dict)
        pr_curves[model_name] = (p_dict, r_dict)

        del item_to_emb, model
        gc.collect()
        torch.cuda.empty_cache()

        all_results[model_name] = metrics
        print("  Done.")

    model_names = list(all_results.keys())
    print("\n\n" + "=" * 70)
    print(f"  RESULTS — {room.upper()} GOLDEN SET")
    print("=" * 70)
    print_section("Triplet metrics",
                  ["triplet_accuracy", "margin_satisfied",
                   "pos_dist_mean", "neg_dist_mean", "dist_gap_mean"],
                  all_results, model_names)
    print_section("Recall@K (per-pair hard negatives)",
                  [f"triplet_recall@{k}" for k in RECALL_K],
                  all_results, model_names)
    print_section("MRR", ["MRR"], all_results, model_names)
    print_section("Scene Recall@K (full gallery)",
                  [f"scene_recall@{k}" for k in SCENE_RECALL_K],
                  all_results, model_names)
    print_section("Precision@K (scene co-occurrence GT)",
                  [f"compat_precision@{k}" for k in PR_K],
                  all_results, model_names)
    print_section("Recall@K    (scene co-occurrence GT)",
                  [f"compat_recall@{k}" for k in PR_K],
                  all_results, model_names)

    json_path = OUTPUT_DIR / f"{room}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[{room}] Results saved → {json_path}")

    plot_distance_distributions(pos_neg_dists,
                                OUTPUT_DIR / f"{room}_distance_distributions.png")
    plot_scene_recall(all_results,
                      OUTPUT_DIR / f"{room}_scene_recall_curves.png")
    plot_precision_recall(pr_curves,
                          OUTPUT_DIR / f"{room}_precision_recall_curves.png")
    plot_key_metrics(all_results,
                     OUTPUT_DIR / f"{room}_key_metrics_comparison.png")
    print(f"[{room}] Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for room in ROOMS:
        run_for_room(room)
    print("\nAll rooms done.")
