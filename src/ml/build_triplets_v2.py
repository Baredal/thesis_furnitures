"""
build_triplets_v3.py — Three-way split: golden / train / val.
                        Percentage-based, balanced across ALL sources.

Golden set:
  - 18% of scenes from EACH source (balanced across deepfurn, sklad_mebliv, wayfair)

Val set:
  - 12% of remaining (non-golden) scenes from EACH source

Train set:
  - Everything else (~70% per source)

Each split generates its own pairs and negatives independently —
no cross-contamination between golden, train, and val.

Usage:
  python src/ml/build_triplets_v3.py
"""

import sys
import json
import csv
import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter

# Import reusable functions from the original build_triplets
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_triplets import (
    load_embeddings,
    _scene_source,
    generate_pairs,
    select_negatives,
    export_embedding_matrix,
    export_triplet_npy,
    TRIPLET_FIELDS,
    NUM_NEGATIVES_PER_PAIR,
    NEG_PERCENTILE_LOW,
    NEG_PERCENTILE_HIGH,
)

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parents[2]
ROOMS_TO_PROCESS = ["bedrooms", "living_rooms"]
RANDOM_SEED = 42

GOLDEN_FRAC = 0.18   # 18% of scenes from each source → golden
VAL_FRAC    = 0.12   # 12% of remaining scenes from each source → val


# ═════════════════════════════════════════════════════════════════════════════
# Three-way split
# ═════════════════════════════════════════════════════════════════════════════

def split_golden_train_val(scene_to_items, golden_frac, val_frac, seed):
    """
    Three-way scene-level split, balanced across ALL sources.

      1. Golden: take `golden_frac` from each source (stratified by category-set)
      2. Val:    take `val_frac` from remaining scenes per source (stratified)
      3. Train:  everything else

    Returns (golden_scenes, train_scenes, val_scenes).
    """
    rng = random.Random(seed)

    # Group scenes by (source, category-set)
    groups: dict[tuple[str, frozenset], list[str]] = defaultdict(list)
    for scene_name, items in scene_to_items.items():
        source = _scene_source(scene_name)
        cat_set = frozenset(it["category"] for it in items)
        groups[(source, cat_set)].append(scene_name)

    golden_scenes: set[str] = set()
    val_scenes: set[str] = set()
    train_scenes: set[str] = set()

    # Process largest groups first for stable splits
    for (source, cat_set), scenes in sorted(groups.items(), key=lambda x: -len(x[1])):
        rng.shuffle(scenes)

        if len(scenes) == 1:
            train_scenes.add(scenes[0])
            continue

        n_golden = max(1, round(len(scenes) * golden_frac))
        n_golden = min(n_golden, len(scenes) - 1)  # leave at least 1 for train/val

        golden_part = scenes[:n_golden]
        rest = scenes[n_golden:]

        n_val = max(1, round(len(rest) * val_frac)) if len(rest) > 1 else 0
        val_part = rest[:n_val]
        train_part = rest[n_val:]

        golden_scenes.update(golden_part)
        val_scenes.update(val_part)
        train_scenes.update(train_part)

    return golden_scenes, train_scenes, val_scenes


# ═════════════════════════════════════════════════════════════════════════════
# Output
# ═════════════════════════════════════════════════════════════════════════════

def save_outputs_v3(train_triplets, val_triplets, golden_triplets,
                    golden_scenes, train_scenes, val_scenes,
                    output_dir, metadata):
    """Save triplets, scene split, and CSVs for all three splits."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON (complete)
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

    # CSVs
    for name, data in [
        ("train_triplets.csv", train_triplets),
        ("val_triplets.csv", val_triplets),
        ("golden_triplets.csv", golden_triplets),
    ]:
        with open(output_dir / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRIPLET_FIELDS)
            writer.writeheader()
            writer.writerows(data)

    # Scene split (standalone)
    with open(output_dir / "scene_split.json", "w", encoding="utf-8") as f:
        json.dump({
            "golden_scenes": sorted(golden_scenes),
            "train_scenes": sorted(train_scenes),
            "val_scenes": sorted(val_scenes),
        }, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
# Visualizations
# ═════════════════════════════════════════════════════════════════════════════

_SPLIT_COLORS = {"train": "#4c72b0", "val": "#55a868", "golden": "#c44e52"}


def plot_visualizations_v3(
    train_triplets, val_triplets, golden_triplets,
    train_scenes, val_scenes, golden_scenes,
    scene_to_items,
    train_sc, val_sc, golden_sc,
    output_dir,
):
    """Generate and save diagnostic plots for the three-way v3 split."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_scenes = train_scenes | val_scenes | golden_scenes
    sources = sorted(set(_scene_source(s) for s in all_scenes))
    categories = sorted({it["category"] for items in scene_to_items.values() for it in items})
    splits = [("train", train_scenes), ("val", val_scenes), ("golden", golden_scenes)]
    split_triplets = {"train": train_triplets, "val": val_triplets, "golden": golden_triplets}
    split_sc = {"train": train_sc, "val": val_sc, "golden": golden_sc}

    src_color_map = {s: c for s, c in zip(sources, ["#4c72b0", "#55a868", "#c44e52", "#8172b2"])}

    # ─── 01. Three-way scene split per source (grouped bar) ───────────────
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

    # ─── 02. Triplet counts summary ───────────────────────────────────────
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

    # ─── 03. Category distribution (items) per split ──────────────────────
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

    # ─── 04. Pair-category heatmaps (train and golden) ────────────────────
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

    # ─── 05. Negative distance distributions (all 3 splits) ──────────────
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

    # ─── 06. Scene negative-usage balance per split ───────────────────────
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

    # ─── 07. Source breakdown of triplet roles (train) ────────────────────
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

    # ─── 08. Per-source split ratio (actual vs target) ────────────────────
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


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def process_room(room_name: str, embeddings_path: Path, output_dir: Path):
    t0 = time.time()

    if not embeddings_path.exists():
        print(f"ERROR: embeddings not found at {embeddings_path}")
        sys.exit(1)

    # ── 1. Load ───────────────────────────────────────────────────────────
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

    # ── 2. Three-way split ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 - Golden / Train / Val split")
    print("=" * 60)
    print(f"  Golden fraction: {GOLDEN_FRAC:.0%}  |  Val fraction: {VAL_FRAC:.0%}")

    golden_scenes, train_scenes, val_scenes = split_golden_train_val(
        scene_to_items,
        golden_frac=GOLDEN_FRAC,
        val_frac=VAL_FRAC,
        seed=RANDOM_SEED,
    )
    print(f"  Train:  {len(train_scenes)} scenes")
    print(f"  Val:    {len(val_scenes)} scenes")
    print(f"  Golden: {len(golden_scenes)} scenes")

    # Per-source breakdown
    all_scenes = train_scenes | val_scenes | golden_scenes
    for src in sorted(set(_scene_source(s) for s in all_scenes)):
        n_tr = sum(1 for s in train_scenes if _scene_source(s) == src)
        n_va = sum(1 for s in val_scenes if _scene_source(s) == src)
        n_go = sum(1 for s in golden_scenes if _scene_source(s) == src)
        total = n_tr + n_va + n_go
        print(f"    {src}: train={n_tr} ({n_tr/total:.0%}), "
              f"val={n_va} ({n_va/total:.0%}), "
              f"golden={n_go} ({n_go/total:.0%}), total={total}")

    # Pair-eligible scenes (2+ categories)
    def multi_cat(ss):
        return {s for s in ss
                if len({it["category"] for it in scene_to_items[s]}) >= 2}

    train_pair = multi_cat(train_scenes)
    val_pair = multi_cat(val_scenes)
    golden_pair = multi_cat(golden_scenes)
    print(f"  Pair-eligible — train: {len(train_pair)}, "
          f"val: {len(val_pair)}, golden: {len(golden_pair)}")

    # ── 3. Pairs ──────────────────────────────────────────────────────────
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

    # ── 4. Negatives ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 - Selecting negatives")
    print("=" * 60)

    print("  [train]")
    train_triplets, train_skip, train_sc = select_negatives(
        train_pairs, scene_to_items, category_to_items, train_scenes,
        NUM_NEGATIVES_PER_PAIR, NEG_PERCENTILE_LOW, NEG_PERCENTILE_HIGH,
        RANDOM_SEED,
    )
    print(f"    Triplets: {len(train_triplets)}  |  skipped: {train_skip}")

    print("  [val]")
    val_triplets, val_skip, val_sc = select_negatives(
        val_pairs, scene_to_items, category_to_items, val_scenes,
        NUM_NEGATIVES_PER_PAIR, NEG_PERCENTILE_LOW, NEG_PERCENTILE_HIGH,
        RANDOM_SEED + 1,
    )
    print(f"    Triplets: {len(val_triplets)}  |  skipped: {val_skip}")

    print("  [golden]")
    golden_triplets, golden_skip, golden_sc = select_negatives(
        golden_pairs, scene_to_items, category_to_items, golden_scenes,
        NUM_NEGATIVES_PER_PAIR, NEG_PERCENTILE_LOW, NEG_PERCENTILE_HIGH,
        RANDOM_SEED + 2,
    )
    print(f"    Triplets: {len(golden_triplets)}  |  skipped: {golden_skip}")

    # ── 5. Save ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 - Saving outputs")
    print("=" * 60)

    metadata = {
        "split_version": "v3_balanced_golden",
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

    save_outputs_v3(
        train_triplets, val_triplets, golden_triplets,
        golden_scenes, train_scenes, val_scenes,
        output_dir, metadata,
    )
    print(f"  Dir: {output_dir}")
    print(f"  Files: triplets.json, scene_split.json, "
          f"train/val/golden_triplets.csv")

    # Embedding matrix + NPY indices
    print("\n  Exporting embedding matrix ...")
    shape = export_embedding_matrix(all_items, output_dir)
    print(f"  embeddings.npz  shape={shape}")

    for name, triplets in [("train", train_triplets),
                           ("val", val_triplets),
                           ("golden", golden_triplets)]:
        npy_path = output_dir / f"{name}_triplet_indices.npy"
        s = export_triplet_npy(triplets, all_items, npy_path)
        print(f"  {npy_path.name}  shape={s}")

    # ── 6. Visualizations ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 - Generating visualizations")
    print("=" * 60)
    plot_visualizations_v3(
        train_triplets, val_triplets, golden_triplets,
        train_scenes, val_scenes, golden_scenes,
        scene_to_items,
        train_sc, val_sc, golden_sc,
        output_dir,
    )

    elapsed = time.time() - t0
    print(f"\nDone ({elapsed:.1f}s)")


def main():
    for room in ROOMS_TO_PROCESS:
        print(f"\n{'#' * 60}")
        print(f"  Processing: {room}")
        print(f"{'#' * 60}\n")

        embeddings_path = (BASE_DIR / "data" / "ml_data" / room
                           / "embeddings" / "furniture_embeddings.json")
        output_dir = BASE_DIR / "data" / "ml_data" / room / "triplets_v3"

        process_room(room, embeddings_path, output_dir)


if __name__ == "__main__":
    main()
