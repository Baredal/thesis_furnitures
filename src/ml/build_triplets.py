import json
import sys
import csv
import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations


BASE_DIR = Path(__file__).resolve().parents[2]
ROOMS_TO_PROCESS = ["bedrooms", "living_rooms"]

NUM_NEGATIVES_PER_PAIR = 5
RANDOM_SEED = 42

# percentile window for negative difficulty: upper half, minus top-5% outliers
NEG_PERCENTILE_LOW = 50
NEG_PERCENTILE_HIGH = 95

GOLDEN_FRAC = 0.18   # reserved for final evaluation only — never touched during training
VAL_FRAC    = 0.12   # used for model selection

TRIPLET_FIELDS = [
    "anchor_id", "anchor_category", "anchor_scene", "anchor_source",
    "positive_id", "positive_category", "positive_scene", "positive_source",
    "negative_id", "negative_category", "negative_scene", "negative_source",
    "pos_neg_distance", "anchor_counterpart_distance", "combined_distance",
]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 1.0
    return float(1.0 - dot / (na * nb))


def load_embeddings(path: Path):
    # one entry per furniture_id; multi-scene membership tracked in item["scenes"]
    print(f"  Reading {path.name} ...")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    all_items: dict = {}
    multi_scene_count = 0

    for source, scenes in raw.items():
        for scene_name, furnitures in scenes.items():
            for furn_id, furn_data in furnitures.items():
                if furn_id in all_items:
                    all_items[furn_id]["scenes"].add(scene_name)
                    multi_scene_count += 1
                else:
                    emb = np.array(furn_data["embedding"], dtype=np.float32)
                    all_items[furn_id] = {
                        "furniture_id": furn_id,
                        "category": furn_data["category"],
                        "source": source,
                        "embedding": emb,
                        "scenes": {scene_name},
                    }

    if multi_scene_count:
        multi_items = sum(1 for it in all_items.values() if len(it["scenes"]) > 1)
        print(f"  {multi_items} items appear in multiple scenes "
              f"({multi_scene_count} extra scene-links)")

    scene_to_items: dict[str, list] = defaultdict(list)
    for item in all_items.values():
        for scene in item["scenes"]:
            scene_to_items[scene].append(item)

    category_to_items: dict[str, list] = defaultdict(list)
    for item in all_items.values():
        category_to_items[item["category"]].append(item)

    return all_items, dict(scene_to_items), dict(category_to_items)


def _scene_source(scene_name: str) -> str:
    for prefix in ("deepfurn", "sklad_mebliv", "wayfair"):  # will match whatever is found
        if scene_name.startswith(prefix):
            return prefix
    return "unknown"


def generate_pairs(scene_to_items: dict, eligible_scenes: set):
    pairs = []
    seen: set[tuple[str, str]] = set()
    same_cat_skipped = 0
    cross_scene_deduped = 0

    for scene_name in sorted(eligible_scenes):
        items = scene_to_items.get(scene_name, [])
        for a, b in combinations(items, 2):
            if a["category"] == b["category"]:
                same_cat_skipped += 1
                continue
            if a["category"] < b["category"]:
                anchor, positive = a, b
            else:
                anchor, positive = b, a

            pair_key = (anchor["furniture_id"], positive["furniture_id"])
            if pair_key in seen:
                cross_scene_deduped += 1
                continue
            seen.add(pair_key)
            pairs.append((anchor, positive, scene_name))

    if same_cat_skipped:
        print(f"    same-category pairs skipped: {same_cat_skipped}")
    if cross_scene_deduped:
        print(f"    cross-scene duplicates removed: {cross_scene_deduped}")

    return pairs


def select_negatives(
    pairs: list,
    scene_to_items: dict,
    category_to_items: dict,
    split_scenes: set,
    num_negatives: int,
    pctile_low: int,
    pctile_high: int,
    seed: int,
):
    rng = random.Random(seed)

    # fast lookup: scene → category → items
    scene_cat_idx: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for sn in split_scenes:
        for it in scene_to_items.get(sn, []):
            scene_cat_idx[sn][it["category"]].append(it)

    scene_neg_count: Counter = Counter()
    total_neg_needed = len(pairs) * num_negatives
    avg_per_scene = total_neg_needed / max(len(split_scenes), 1)
    max_per_scene = max(num_negatives * 3, int(avg_per_scene * 5))

    triplets: list[dict] = []
    skipped = 0

    # shuffle pair order to avoid systematic scene bias
    pair_order = list(range(len(pairs)))
    rng.shuffle(pair_order)

    for idx in pair_order:
        anchor, positive, pair_scene = pairs[idx]
        anchor_cat = anchor["category"]
        pos_cat = positive["category"]
        anchor_emb = anchor["embedding"]
        pos_emb = positive["embedding"]

        # exclude all scenes where anchor or positive appear
        excluded_scenes = anchor["scenes"] | positive["scenes"]

        cands_with_cp: list[dict] = []
        cands_no_cp: list[dict] = []

        for neg_item in category_to_items.get(pos_cat, []):
            if neg_item["scenes"] & excluded_scenes:
                continue
            neg_valid_scenes = neg_item["scenes"] & split_scenes
            if not neg_valid_scenes:
                continue

            d_pos_neg = cosine_distance(pos_emb, neg_item["embedding"])

            best_cp_dist = None
            best_cp_scene = None
            for ns in neg_valid_scenes:
                for cp in scene_cat_idx[ns].get(anchor_cat, []):
                    if cp["furniture_id"] == anchor["furniture_id"]:
                        continue
                    d = cosine_distance(anchor_emb, cp["embedding"])
                    if best_cp_dist is None or d < best_cp_dist:
                        best_cp_dist = d
                        best_cp_scene = ns

            if best_cp_dist is not None:
                cands_with_cp.append({
                    "neg_item": neg_item,
                    "neg_scene": best_cp_scene,
                    "d_pos_neg": d_pos_neg,
                    "d_anchor_cp": best_cp_dist,
                    "combined": d_pos_neg + best_cp_dist,
                })
            else:
                cands_no_cp.append({
                    "neg_item": neg_item,
                    "neg_scene": sorted(neg_valid_scenes)[0],
                    "d_pos_neg": d_pos_neg,
                    "d_anchor_cp": None,
                    "combined": None,
                })

        # fill missing counterpart distances with median so they can be ranked
        if cands_with_cp:
            median_cp = float(np.median([c["d_anchor_cp"] for c in cands_with_cp]))
        else:
            median_cp = 0.5
        for c in cands_no_cp:
            c["d_anchor_cp"] = median_cp
            c["combined"] = c["d_pos_neg"] + median_cp

        all_cands = cands_with_cp + cands_no_cp
        if not all_cands:
            skipped += 1
            continue

        all_cands.sort(key=lambda c: c["combined"])
        n = len(all_cands)
        lo = int(n * pctile_low / 100)
        hi = max(lo + 1, int(n * pctile_high / 100))
        eligible = all_cands[lo:hi]
        if not eligible:
            eligible = all_cands

        rng.shuffle(eligible)
        eligible.sort(key=lambda c: scene_neg_count[c["neg_scene"]])

        selected: list[dict] = []
        selected_ids: set[str] = set()
        used_scenes: set[str] = set()

        # pass 1: one negative per scene, respect quota
        for cand in eligible:
            if len(selected) >= num_negatives:
                break
            nid = cand["neg_item"]["furniture_id"]
            ns = cand["neg_scene"]
            if ns not in used_scenes and nid not in selected_ids and scene_neg_count[ns] < max_per_scene:
                selected.append(cand)
                selected_ids.add(nid)
                used_scenes.add(ns)
                scene_neg_count[ns] += 1

        # pass 2: relax scene-uniqueness if still short
        if len(selected) < num_negatives:
            for cand in eligible:
                if len(selected) >= num_negatives:
                    break
                nid = cand["neg_item"]["furniture_id"]
                if nid not in selected_ids:
                    selected.append(cand)
                    selected_ids.add(nid)
                    scene_neg_count[cand["neg_scene"]] += 1

        for neg in selected:
            triplets.append({
                "anchor_id": anchor["furniture_id"],
                "anchor_category": anchor_cat,
                "anchor_scene": pair_scene,
                "anchor_source": anchor["source"],
                "positive_id": positive["furniture_id"],
                "positive_category": pos_cat,
                "positive_scene": pair_scene,
                "positive_source": positive["source"],
                "negative_id": neg["neg_item"]["furniture_id"],
                "negative_category": neg["neg_item"]["category"],
                "negative_scene": neg["neg_scene"],
                "negative_source": neg["neg_item"]["source"],
                "pos_neg_distance": round(neg["d_pos_neg"], 6),
                "anchor_counterpart_distance": round(neg["d_anchor_cp"], 6),
                "combined_distance": round(neg["combined"], 6),
            })

    return triplets, skipped, dict(scene_neg_count)


def export_embedding_matrix(all_items: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    furn_ids = sorted(all_items.keys())
    id_to_idx = {fid: i for i, fid in enumerate(furn_ids)}

    matrix = np.stack([all_items[fid]["embedding"] for fid in furn_ids])
    np.savez_compressed(output_dir / "embeddings.npz", embeddings=matrix)

    mapping = {
        "id_to_index": id_to_idx,
        "index_to_id": {str(i): fid for fid, i in id_to_idx.items()},
        "embedding_dim": int(matrix.shape[1]),
        "num_items": int(matrix.shape[0]),
    }
    with open(output_dir / "embedding_index.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    return matrix.shape


def export_triplet_npy(triplets: list[dict], all_items: dict, output_path: Path):
    furn_ids = sorted(all_items.keys())
    id_to_idx = {fid: i for i, fid in enumerate(furn_ids)}

    indices = np.empty((len(triplets), 3), dtype=np.int32)
    for i, t in enumerate(triplets):
        indices[i, 0] = id_to_idx[t["anchor_id"]]
        indices[i, 1] = id_to_idx[t["positive_id"]]
        indices[i, 2] = id_to_idx[t["negative_id"]]

    np.save(output_path, indices)
    return indices.shape


_SPLIT_COLORS = {"train": "#4c72b0", "val": "#55a868", "golden": "#c44e52"}


def split_golden_train_val(scene_to_items, golden_frac, val_frac, seed):
    rng = random.Random(seed)

    groups: dict[tuple[str, frozenset], list[str]] = defaultdict(list)
    for scene_name, items in scene_to_items.items():
        source = _scene_source(scene_name)
        cat_set = frozenset(it["category"] for it in items)
        groups[(source, cat_set)].append(scene_name)

    golden_scenes: set[str] = set()
    val_scenes: set[str] = set()
    train_scenes: set[str] = set()

    for (source, cat_set), scenes in sorted(groups.items(), key=lambda x: -len(x[1])):
        rng.shuffle(scenes)

        if len(scenes) == 1:
            train_scenes.add(scenes[0])
            continue

        n_golden = max(1, round(len(scenes) * golden_frac))
        n_golden = min(n_golden, len(scenes) - 1)

        golden_part = scenes[:n_golden]
        rest = scenes[n_golden:]

        n_val = max(1, round(len(rest) * val_frac)) if len(rest) > 1 else 0
        val_part = rest[:n_val]
        train_part = rest[n_val:]

        golden_scenes.update(golden_part)
        val_scenes.update(val_part)
        train_scenes.update(train_part)

    return golden_scenes, train_scenes, val_scenes


def _save_triplets(train_triplets, val_triplets, golden_triplets,
                   golden_scenes, train_scenes, val_scenes,
                   output_dir, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "triplets.json", "w", encoding="utf-8") as f:
        json.dump({
            "metadata": metadata,
            "golden_scenes": sorted(golden_scenes),
            "train_scenes": sorted(train_scenes),
            "val_scenes": sorted(val_scenes),
            "train_triplets": train_triplets,
            "val_triplets": val_triplets,
            "golden_triplets": golden_triplets,
        }, f, indent=2)

    for name, data in [
        ("train_triplets.csv", train_triplets),
        ("val_triplets.csv", val_triplets),
        ("golden_triplets.csv", golden_triplets),
    ]:
        with open(output_dir / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRIPLET_FIELDS)
            writer.writeheader()
            writer.writerows(data)

    with open(output_dir / "scene_split.json", "w", encoding="utf-8") as f:
        json.dump({
            "golden_scenes": sorted(golden_scenes),
            "train_scenes": sorted(train_scenes),
            "val_scenes": sorted(val_scenes),
        }, f, indent=2)


def _plot_triplets(train_triplets, val_triplets, golden_triplets,
                   train_scenes, val_scenes, golden_scenes,
                   scene_to_items, train_sc, val_sc, golden_sc,
                   output_dir):
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_scenes = train_scenes | val_scenes | golden_scenes
    sources = sorted(set(_scene_source(s) for s in all_scenes))
    categories = sorted({it["category"] for items in scene_to_items.values() for it in items})
    splits = [("train", train_scenes), ("val", val_scenes), ("golden", golden_scenes)]
    split_triplets = {"train": train_triplets, "val": val_triplets, "golden": golden_triplets}
    split_sc = {"train": train_sc, "val": val_sc, "golden": golden_sc}

    src_color_map = {s: c for s, c in zip(sources, ["#4c72b0", "#55a868", "#c44e52", "#8172b2"])}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sources))
    width = 0.25
    for i, (split_name, ss) in enumerate(splits):
        counts = [sum(1 for sc in ss if _scene_source(sc) == src) for src in sources]
        bars = ax.bar(x + i * width, counts, width, label=split_name,
                      color=_SPLIT_COLORS[split_name])
        for bar, v in zip(bars, counts):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, str(v),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x + width)
    ax.set_xticklabels(sources)
    ax.set_ylabel("Scenes")
    ax.set_title(f"Three-way scene split per source  (golden={GOLDEN_FRAC:.0%}, val={VAL_FRAC:.0%})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "01_threeway_split_by_source.png", dpi=150)
    plt.close(fig)

    triplet_counts = [len(t) for _, t in
                      [("train", train_triplets), ("val", val_triplets), ("golden", golden_triplets)]]
    split_labels = ["train", "val", "golden"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(split_labels, triplet_counts,
                  color=[_SPLIT_COLORS[s] for s in split_labels])
    for bar, v in zip(bars, triplet_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(triplet_counts) * 0.01,
                f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Triplets")
    ax.set_title("Triplet counts per split")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_triplet_counts.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, (split_name, ss) in zip(axes, splits):
        cc = Counter(it["category"] for s in ss for it in scene_to_items[s])
        vals = [cc.get(c, 0) for c in categories]
        bars = ax.bar(categories, vals, color=_SPLIT_COLORS[split_name])
        ax.set_title(f"{split_name} — items by category")
        ax.set_ylabel("Items")
        ax.tick_params(axis="x", rotation=35)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, str(v),
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Category distribution per split", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_category_distribution.png", dpi=150)
    plt.close(fig)

    for split_name, triplets in [("train", train_triplets), ("golden", golden_triplets)]:
        if not triplets:
            continue
        pair_counts = Counter(
            (t["anchor_category"], t["positive_category"]) for t in triplets
        )
        mat = np.zeros((len(categories), len(categories)))
        for (ac, pc), cnt in pair_counts.items():
            if ac in categories and pc in categories:
                i, j = categories.index(ac), categories.index(pc)
                mat[i, j] = cnt
                mat[j, i] = cnt
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(mat, cmap="YlOrRd")
        ax.set_xticks(range(len(categories)))
        ax.set_yticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_yticklabels(categories)
        ax.set_title(f"{split_name} triplets: anchor–positive category pairs")
        for i in range(len(categories)):
            for j in range(len(categories)):
                if mat[i, j] > 0:
                    ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center",
                            fontsize=8,
                            color="white" if mat[i, j] > mat.max() * 0.6 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"04_pair_heatmap_{split_name}.png", dpi=150)
        plt.close(fig)

    dist_keys = [
        ("pos_neg_distance", "Positive ↔ Negative"),
        ("anchor_counterpart_distance", "Anchor ↔ Neg-scene counterpart"),
        ("combined_distance", "Combined score"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (dist_key, title) in zip(axes, dist_keys):
        for split_name, triplets in split_triplets.items():
            vals = [t[dist_key] for t in triplets if t.get(dist_key) is not None]
            if vals:
                ax.hist(vals, bins=50, alpha=0.55, label=split_name,
                        color=_SPLIT_COLORS[split_name])
        ax.set_title(title)
        ax.set_xlabel("Cosine distance")
        ax.set_ylabel("Count")
        ax.legend()
    fig.suptitle("Negative distance distributions (train / val / golden)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "05_distance_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (split_name, ss) in zip(axes, splits):
        sc = split_sc[split_name]
        if not ss:
            continue
        all_counts = [sc.get(scene, 0) for scene in ss]
        ax.hist(all_counts, bins=30, color=_SPLIT_COLORS[split_name], edgecolor="white")
        mean_val = np.mean(all_counts) if all_counts else 0
        median_val = np.median(all_counts) if all_counts else 0
        zero_scenes = sum(1 for v in all_counts if v == 0)
        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color="blue", linestyle="--", label=f"Median: {median_val:.0f}")
        textstr = (f"Scenes: {len(ss)}\n"
                   f"Never used: {zero_scenes}\n"
                   f"Max used: {max(all_counts) if all_counts else 0}")
        ax.text(0.62, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.set_title(f"{split_name} — negative scene usage")
        ax.set_xlabel("Negatives drawn from a scene")
        ax.set_ylabel("Number of scenes")
        ax.legend(fontsize=8)
    fig.suptitle("Negative sampling balance per split", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "06_scene_neg_balance.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (role, key) in zip(axes, [
        ("Anchor", "anchor_source"),
        ("Positive", "positive_source"),
        ("Negative", "negative_source"),
    ]):
        cc = Counter(t[key] for t in train_triplets)
        vals = [cc.get(s, 0) for s in sources]
        bars = ax.bar(sources, vals, color=[src_color_map[s] for s in sources])
        ax.set_title(f"Train triplets: {role} source")
        ax.set_ylabel("Count")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals, default=1) * 0.01,
                    str(v), ha="center", va="bottom", fontsize=9)
    fig.suptitle("Source breakdown of triplet roles (train split)", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "07_triplet_source_roles.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sources))
    width = 0.25
    for i, (split_name, ss) in enumerate(splits):
        ratios = []
        for src in sources:
            src_total = sum(1 for sc in all_scenes if _scene_source(sc) == src)
            src_in_split = sum(1 for sc in ss if _scene_source(sc) == src)
            ratios.append(src_in_split / src_total if src_total > 0 else 0)
        bars = ax.bar(x + i * width, ratios, width, label=split_name,
                      color=_SPLIT_COLORS[split_name])
        for bar, v in zip(bars, ratios):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=8)
    ax.axhline(GOLDEN_FRAC, color="#c44e52", linestyle="--", linewidth=1,
               label=f"Golden target {GOLDEN_FRAC:.0%}")
    ax.axhline(VAL_FRAC, color="#55a868", linestyle="--", linewidth=1,
               label=f"Val target {VAL_FRAC:.0%}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(sources)
    ax.set_ylabel("Fraction of source scenes")
    ax.set_title("Actual split ratio per source vs targets")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "08_per_source_split_ratio.png", dpi=150)
    plt.close(fig)

    print(f"  Figures saved to {fig_dir}  (8 plots)")


def process_room(room_name: str, embeddings_path: Path, output_dir: Path):
    t0 = time.time()

    if not embeddings_path.exists():
        print(f"ERROR: embeddings not found at {embeddings_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"STEP 1 - Loading embeddings ({room_name})")
    print("=" * 60)
    all_items, scene_to_items, category_to_items = load_embeddings(embeddings_path)
    cats = sorted(category_to_items.keys())
    print(f"  Unique items: {len(all_items)}")
    print(f"  Scenes:       {len(scene_to_items)}")
    print(f"  Categories:   {cats}")
    for c in cats:
        print(f"    {c}: {len(category_to_items[c])} items")

    print("\n" + "=" * 60)
    print("STEP 2 - Golden / Train / Val split")
    print("=" * 60)
    print(f"  Golden fraction: {GOLDEN_FRAC:.0%}  |  Val fraction: {VAL_FRAC:.0%}")

    golden_scenes, train_scenes, val_scenes = split_golden_train_val(
        scene_to_items, golden_frac=GOLDEN_FRAC, val_frac=VAL_FRAC, seed=RANDOM_SEED,
    )
    print(f"  Train:  {len(train_scenes)} scenes")
    print(f"  Val:    {len(val_scenes)} scenes")
    print(f"  Golden: {len(golden_scenes)} scenes")

    all_scenes = train_scenes | val_scenes | golden_scenes
    for src in sorted(set(_scene_source(s) for s in all_scenes)):
        n_tr = sum(1 for s in train_scenes if _scene_source(s) == src)
        n_va = sum(1 for s in val_scenes if _scene_source(s) == src)
        n_go = sum(1 for s in golden_scenes if _scene_source(s) == src)
        total = n_tr + n_va + n_go
        print(f"    {src}: train={n_tr} ({n_tr/total:.0%}), "
              f"val={n_va} ({n_va/total:.0%}), "
              f"golden={n_go} ({n_go/total:.0%}), total={total}")

    def multi_cat(ss):
        return {s for s in ss if len({it["category"] for it in scene_to_items[s]}) >= 2}

    train_pair = multi_cat(train_scenes)
    val_pair = multi_cat(val_scenes)
    golden_pair = multi_cat(golden_scenes)
    print(f"  Pair-eligible — train: {len(train_pair)}, "
          f"val: {len(val_pair)}, golden: {len(golden_pair)}")

    print("\n" + "=" * 60)
    print("STEP 3 - Generating anchor-positive pairs")
    print("=" * 60)
    print("  [train]")
    train_pairs = generate_pairs(scene_to_items, train_pair)
    print(f"    Pairs: {len(train_pairs)}")

    print("  [val]")
    val_pairs = generate_pairs(scene_to_items, val_pair)
    print(f"    Pairs: {len(val_pairs)}")

    print("  [golden]")
    golden_pairs = generate_pairs(scene_to_items, golden_pair)
    print(f"    Pairs: {len(golden_pairs)}")

    print("\n" + "=" * 60)
    print("STEP 4 - Selecting negatives")
    print("=" * 60)

    print("  [train]")
    train_triplets, train_skip, train_sc = select_negatives(
        train_pairs, scene_to_items, category_to_items, train_scenes,
        NUM_NEGATIVES_PER_PAIR, NEG_PERCENTILE_LOW, NEG_PERCENTILE_HIGH, RANDOM_SEED,
    )
    print(f"    Triplets: {len(train_triplets)}  |  skipped: {train_skip}")

    print("  [val]")
    val_triplets, val_skip, val_sc = select_negatives(
        val_pairs, scene_to_items, category_to_items, val_scenes,
        NUM_NEGATIVES_PER_PAIR, NEG_PERCENTILE_LOW, NEG_PERCENTILE_HIGH, RANDOM_SEED + 1,
    )
    print(f"    Triplets: {len(val_triplets)}  |  skipped: {val_skip}")

    print("  [golden]")
    golden_triplets, golden_skip, golden_sc = select_negatives(
        golden_pairs, scene_to_items, category_to_items, golden_scenes,
        NUM_NEGATIVES_PER_PAIR, NEG_PERCENTILE_LOW, NEG_PERCENTILE_HIGH, RANDOM_SEED + 2,
    )
    print(f"    Triplets: {len(golden_triplets)}  |  skipped: {golden_skip}")

    print("\n" + "=" * 60)
    print("STEP 5 - Saving outputs")
    print("=" * 60)

    metadata = {
        "split_version": "balanced_golden",
        "golden_frac": GOLDEN_FRAC,
        "val_frac": VAL_FRAC,
        "num_negatives_per_pair": NUM_NEGATIVES_PER_PAIR,
        "neg_percentile_range": [NEG_PERCENTILE_LOW, NEG_PERCENTILE_HIGH],
        "random_seed": RANDOM_SEED,
        "total_items": len(all_items),
        "total_scenes": len(scene_to_items),
        "categories": cats,
        "num_golden_scenes": len(golden_scenes),
        "num_train_scenes": len(train_scenes),
        "num_val_scenes": len(val_scenes),
        "num_golden_pair_scenes": len(golden_pair),
        "num_train_pair_scenes": len(train_pair),
        "num_val_pair_scenes": len(val_pair),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "num_golden_pairs": len(golden_pairs),
        "num_train_triplets": len(train_triplets),
        "num_val_triplets": len(val_triplets),
        "num_golden_triplets": len(golden_triplets),
        "train_skipped_pairs": train_skip,
        "val_skipped_pairs": val_skip,
        "golden_skipped_pairs": golden_skip,
    }

    _save_triplets(
        train_triplets, val_triplets, golden_triplets,
        golden_scenes, train_scenes, val_scenes,
        output_dir, metadata,
    )
    print(f"  Dir: {output_dir}")
    print(f"  Files: triplets.json, scene_split.json, train/val/golden_triplets.csv")

    print("\n  Exporting embedding matrix ...")
    shape = export_embedding_matrix(all_items, output_dir)
    print(f"  embeddings.npz  shape={shape}")

    for name, triplets in [("train", train_triplets), ("val", val_triplets), ("golden", golden_triplets)]:
        npy_path = output_dir / f"{name}_triplet_indices.npy"
        s = export_triplet_npy(triplets, all_items, npy_path)
        print(f"  {npy_path.name}  shape={s}")

    print("\n" + "=" * 60)
    print("STEP 6 - Generating visualizations")
    print("=" * 60)
    _plot_triplets(
        train_triplets, val_triplets, golden_triplets,
        train_scenes, val_scenes, golden_scenes,
        scene_to_items, train_sc, val_sc, golden_sc,
        output_dir,
    )

    elapsed = time.time() - t0
    print(f"\nDone ({elapsed:.1f}s)")


def main():
    for room in ROOMS_TO_PROCESS:
        emb_path = BASE_DIR / "data" / "ml_data" / room / "embeddings" / "furniture_embeddings.json"
        if not emb_path.exists():
            print(f"Skipping {room}: no embeddings at {emb_path}")
            continue

        out_dir = BASE_DIR / "data" / "ml_data" / room / "triplets_v3"
        print(f"\n{'#' * 60}")
        print(f"  Processing: {room}")
        print(f"{'#' * 60}\n")
        process_room(room, emb_path, out_dir)


if __name__ == "__main__":
    main()
