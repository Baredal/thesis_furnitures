"""
build_triplets.py — Construct triplet-loss training data for furniture compatibility.

Triplet structure:
  Anchor:   furniture item from a scene
  Positive: different-category item from the SAME scene (compatible pair)
  Negative: same category as positive, from a DIFFERENT scene,
            selected by dual cosine-distance criterion

Negative selection strategy:
  1. Candidate must be the same category as the positive
  2. Candidate must NOT share ANY scene with anchor or positive
     (same item can appear in multiple scenes → all are excluded)
  3. Scored by dual criterion:
       combined = cos_dist(positive, negative)
                + cos_dist(anchor, closest_anchor-cat_counterpart_in_neg_scene)
     This ensures the negative is genuinely different from the positive AND comes from
     a stylistically different room (anchor counterpart is also different).
  4. Selected from the [50th, 95th] percentile of combined scores
     (moderately-to-very different, excluding extreme outliers)
  5. Balanced across source scenes using per-scene quotas

Pair deduplication:
  (bed, nightstand) and (nightstand, bed) are the same pair — kept only once
  via canonical ordering (anchor_category < positive_category alphabetically).
  If the exact same two items co-occur in multiple scenes, only one pair is emitted.

Train/test split:
  - Scene-level split (no furniture leakage)
  - Stratified by category composition for diverse test coverage
  - ~15% scenes reserved for test

Usage:
  python src/ml/build_triplets.py
"""

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


# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parents[2]
ROOMS_TO_PROCESS = ["bedrooms", "living_rooms"]

TEST_RATIO = 0.15
NUM_NEGATIVES_PER_PAIR = 5
RANDOM_SEED = 42

# Percentile window for "relatively different" negatives.
# 50–95 means: upper half of distance distribution, minus top-5% outliers.
NEG_PERCENTILE_LOW = 50
NEG_PERCENTILE_HIGH = 95


# ═════════════════════════════════════════════════════════════════════════════
# Distance
# ═════════════════════════════════════════════════════════════════════════════

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity.  Range [0, 1] for non-negative (ReLU) features."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 1.0
    return float(1.0 - dot / (na * nb))


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_embeddings(path: Path):
    """
    Load the nested embeddings JSON.  The same furniture_id can appear in
    multiple scenes — we keep ONE item per furniture_id but track ALL scenes
    it belongs to in item["scenes"] (set).

    Returns:
      all_items         — furniture_id → item dict  (with "scenes": set)
      scene_to_items    — scene_name   → [item, …]  (items are shared objects)
      category_to_items — category     → [item, …]  (unique by furniture_id)
    """
    print(f"  Reading {path.name} …")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    all_items: dict = {}
    multi_scene_count = 0

    for source, scenes in raw.items():
        for scene_name, furnitures in scenes.items():
            for furn_id, furn_data in furnitures.items():
                if furn_id in all_items:
                    # Same furniture appearing in another scene — just add scene
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

    # Build scene → items  (each item may appear in several scene lists)
    scene_to_items: dict[str, list] = defaultdict(list)
    for item in all_items.values():
        for scene in item["scenes"]:
            scene_to_items[scene].append(item)

    # Build category → items  (one entry per unique furniture_id)
    category_to_items: dict[str, list] = defaultdict(list)
    for item in all_items.values():
        category_to_items[item["category"]].append(item)

    return all_items, dict(scene_to_items), dict(category_to_items)


# ═════════════════════════════════════════════════════════════════════════════
# Train / test split
# ═════════════════════════════════════════════════════════════════════════════

def _scene_source(scene_name: str) -> str:
    """Extract data source from scene name (e.g. 'deepfurn_0042' → 'deepfurn')."""
    for prefix in ("deepfurn", "sklad_mebliv", "wayfair"): # It will match whatever
        if scene_name.startswith(prefix):
            return prefix
    return "unknown"


def split_train_test(scene_to_items: dict, test_ratio: float, seed: int):
    """
    Scene-level stratified split, balanced across data sources.

    Scenes are grouped by (source, category-set). Within each group,
    ~test_ratio scenes are sampled for test. This guarantees:
      1) ~15% from each source (deepfurn, wayfair, sklad_mebliv)
      2) Diverse category coverage within each source
      3) No furniture leakage (scene-level split)
    """
    rng = random.Random(seed)

    # Group by (source, category-set)
    groups: dict[tuple[str, frozenset], list[str]] = defaultdict(list)
    for scene_name, items in scene_to_items.items():
        source = _scene_source(scene_name)
        cat_set = frozenset(it["category"] for it in items)
        groups[(source, cat_set)].append(scene_name)

    train_scenes: set[str] = set()
    test_scenes: set[str] = set()

    # Process largest groups first for more stable splits
    for key, scenes in sorted(groups.items(), key=lambda x: -len(x[1])):
        rng.shuffle(scenes)
        n_test = max(1, round(len(scenes) * test_ratio))
        if len(scenes) == 1:
            train_scenes.add(scenes[0])
            continue
        test_scenes.update(scenes[:n_test])
        train_scenes.update(scenes[n_test:])

    return train_scenes, test_scenes


# ═════════════════════════════════════════════════════════════════════════════
# Anchor-positive pair generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_pairs(scene_to_items: dict, eligible_scenes: set):
    """
    For each eligible scene, enumerate all item pairs with DIFFERENT categories.
    Canonical order: category(anchor) < category(positive) alphabetically.

    Deduplication: if the exact same (anchor_id, positive_id) co-occur in
    multiple scenes, only the first-encountered pair is kept.
    """
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


# ═════════════════════════════════════════════════════════════════════════════
# Negative selection
# ═════════════════════════════════════════════════════════════════════════════

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
    """
    For every (anchor, positive, pair_scene) triple, select up to
    `num_negatives` negatives.

    Scoring (dual cosine distance):
      combined = cos_dist(positive, negative)
               + cos_dist(anchor, best_anchor-cat_counterpart_in_neg_scene)

    CRITICAL: a candidate is excluded if it shares ANY scene with the anchor
    or positive (prevents false negatives from shared-room items).

    Selection window: [pctile_low, pctile_high] of sorted combined scores.
    Balance: per-scene quota caps how many times any single scene provides negatives.
    """
    rng = random.Random(seed)

    # Fast look-up: scene → category → [items]
    scene_cat_idx: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for sn in split_scenes:
        for it in scene_to_items.get(sn, []):
            scene_cat_idx[sn][it["category"]].append(it)

    # Scene-balance tracking
    scene_neg_count: Counter = Counter()
    total_neg_needed = len(pairs) * num_negatives
    avg_per_scene = total_neg_needed / max(len(split_scenes), 1)
    max_per_scene = max(num_negatives * 3, int(avg_per_scene * 5))

    triplets: list[dict] = []
    skipped = 0

    # Randomise processing order so no systematic scene-bias
    pair_order = list(range(len(pairs)))
    rng.shuffle(pair_order)

    for idx in pair_order:
        anchor, positive, pair_scene = pairs[idx]
        anchor_cat = anchor["category"]
        pos_cat = positive["category"]
        anchor_emb = anchor["embedding"]
        pos_emb = positive["embedding"]

        # ALL scenes where anchor OR positive appear — these are excluded
        excluded_scenes = anchor["scenes"] | positive["scenes"]

        # ── gather candidates ──
        cands_with_cp: list[dict] = []
        cands_no_cp: list[dict] = []

        for neg_item in category_to_items.get(pos_cat, []):
            # Skip if negative shares ANY scene with anchor/positive
            if neg_item["scenes"] & excluded_scenes:
                continue
            # Skip if negative has no scenes in this split
            neg_valid_scenes = neg_item["scenes"] & split_scenes
            if not neg_valid_scenes:
                continue

            d_pos_neg = cosine_distance(pos_emb, neg_item["embedding"])

            # Counterpart check: across ALL valid scenes of the negative,
            # find the closest anchor-category counterpart.
            best_cp_dist = None
            best_cp_scene = None
            for ns in neg_valid_scenes:
                for cp in scene_cat_idx[ns].get(anchor_cat, []):
                    if cp["furniture_id"] == anchor["furniture_id"]:
                        continue  # should not happen (excluded) but guard
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
                # No anchor-category counterpart in any of negative's scenes
                cands_no_cp.append({
                    "neg_item": neg_item,
                    "neg_scene": sorted(neg_valid_scenes)[0],
                    "d_pos_neg": d_pos_neg,
                    "d_anchor_cp": None,
                    "combined": None,
                })

        # Fill missing counterpart distances with median from known ones
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

        # ── percentile window ──
        all_cands.sort(key=lambda c: c["combined"])
        n = len(all_cands)
        lo = int(n * pctile_low / 100)
        hi = max(lo + 1, int(n * pctile_high / 100))
        eligible = all_cands[lo:hi]
        if not eligible:
            eligible = all_cands

        # ── balanced selection ──
        rng.shuffle(eligible)
        eligible.sort(key=lambda c: scene_neg_count[c["neg_scene"]])

        selected: list[dict] = []
        selected_ids: set[str] = set()
        used_scenes: set[str] = set()

        # Pass 1: one negative per scene, respect quota
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

        # Pass 2: relax scene-uniqueness if still short
        if len(selected) < num_negatives:
            for cand in eligible:
                if len(selected) >= num_negatives:
                    break
                nid = cand["neg_item"]["furniture_id"]
                if nid not in selected_ids:
                    selected.append(cand)
                    selected_ids.add(nid)
                    scene_neg_count[cand["neg_scene"]] += 1

        # ── emit triplets ──
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


# ═════════════════════════════════════════════════════════════════════════════
# Output
# ═════════════════════════════════════════════════════════════════════════════

TRIPLET_FIELDS = [
    "anchor_id", "anchor_category", "anchor_scene", "anchor_source",
    "positive_id", "positive_category", "positive_scene", "positive_source",
    "negative_id", "negative_category", "negative_scene", "negative_source",
    "pos_neg_distance", "anchor_counterpart_distance", "combined_distance",
]


def save_outputs(
    train_triplets, test_triplets, train_scenes, test_scenes, output_dir, metadata
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON (complete, self-contained) ──
    with open(output_dir / "triplets.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "train_scenes": sorted(train_scenes),
                "test_scenes": sorted(test_scenes),
                "train_triplets": train_triplets,
                "test_triplets": test_triplets,
            },
            f,
            indent=2,
        )

    # ── CSV (flat, easy to load with pandas for remote training) ──
    for name, data in [
        ("train_triplets.csv", train_triplets),
        ("test_triplets.csv", test_triplets),
    ]:
        with open(output_dir / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRIPLET_FIELDS)
            writer.writeheader()
            writer.writerows(data)

    # ── Scene split (standalone, small) ──
    with open(output_dir / "scene_split.json", "w", encoding="utf-8") as f:
        json.dump(
            {"train_scenes": sorted(train_scenes), "test_scenes": sorted(test_scenes)},
            f,
            indent=2,
        )


def export_embedding_matrix(all_items: dict, output_dir: Path):
    """
    Export embeddings as a compact .npz matrix + a JSON id→index mapping.
    Much faster to load during training than the original 50 MB+ JSON.

    Files produced:
      embeddings.npz        — numpy array  shape (N, 2048)
      embedding_index.json  — {"id_to_index": {furn_id: row}, …}
    """
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
    """
    Save triplets as a (T, 3) int32 numpy array of embedding-matrix indices.
    Column order: [anchor_idx, positive_idx, negative_idx].
    Load with: indices = np.load('train_triplet_indices.npy').
    """
    # Build furniture_id → row-index (same order as export_embedding_matrix)
    furn_ids = sorted(all_items.keys())
    id_to_idx = {fid: i for i, fid in enumerate(furn_ids)}

    indices = np.empty((len(triplets), 3), dtype=np.int32)
    for i, t in enumerate(triplets):
        indices[i, 0] = id_to_idx[t["anchor_id"]]
        indices[i, 1] = id_to_idx[t["positive_id"]]
        indices[i, 2] = id_to_idx[t["negative_id"]]

    np.save(output_path, indices)
    return indices.shape


# ═════════════════════════════════════════════════════════════════════════════
# Visualizations
# ═════════════════════════════════════════════════════════════════════════════

def plot_visualizations(
    train_triplets, test_triplets,
    train_scenes, test_scenes,
    scene_to_items, train_sc, test_sc,
    output_dir,
):
    """Generate and save diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    sources = list(set(_scene_source(s) for s in train_scenes | test_scenes))
    categories = sorted({it["category"] for items in scene_to_items.values() for it in items})

    # ─── 1. Source distribution: train vs test scenes ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, ss) in zip(axes, [("Train", train_scenes), ("Test", test_scenes)]):
        src_counts = Counter(_scene_source(s) for s in ss)
        vals = [src_counts.get(s, 0) for s in sources]
        bars = ax.bar(sources, vals, color=["#4c72b0", "#55a868", "#c44e52"])
        ax.set_title(f"{label} scenes by source")
        ax.set_ylabel("Scenes")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v),
                    ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_source_distribution.png", dpi=150)
    plt.close(fig)

    # ─── 2. Category distribution: train vs test items ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (label, ss) in zip(axes, [("Train", train_scenes), ("Test", test_scenes)]):
        cc = Counter()
        for s in ss:
            for it in scene_to_items[s]:
                cc[it["category"]] += 1
        vals = [cc.get(c, 0) for c in categories]
        bars = ax.bar(categories, vals, color="#4c72b0")
        ax.set_title(f"{label} items by category")
        ax.set_ylabel("Items")
        ax.tick_params(axis="x", rotation=30)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_category_distribution.png", dpi=150)
    plt.close(fig)

    # ─── 3. Pair-category heatmap (train) ─────────────────────────────────
    pair_counts = Counter(
        (t["anchor_category"], t["positive_category"]) for t in train_triplets
    )
    mat = np.zeros((len(categories), len(categories)))
    for (ac, pc), cnt in pair_counts.items():
        i, j = categories.index(ac), categories.index(pc)
        mat[i, j] = cnt
        mat[j, i] = cnt  # symmetric display
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="YlOrRd")
    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticklabels(categories)
    ax.set_title("Train triplets: anchor–positive category pairs")
    for i in range(len(categories)):
        for j in range(len(categories)):
            if mat[i, j] > 0:
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center",
                        fontsize=8, color="white" if mat[i, j] > mat.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_pair_category_heatmap.png", dpi=150)
    plt.close(fig)

    # ─── 4. Negative distance distributions ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (dist_key, title) in zip(axes, [
        ("pos_neg_distance", "Positive ↔ Negative"),
        ("anchor_counterpart_distance", "Anchor ↔ Neg-scene counterpart"),
        ("combined_distance", "Combined score"),
    ]):
        train_vals = [t[dist_key] for t in train_triplets]
        test_vals = [t[dist_key] for t in test_triplets]
        ax.hist(train_vals, bins=50, alpha=0.6, label="train", color="#4c72b0")
        ax.hist(test_vals, bins=50, alpha=0.6, label="test", color="#c44e52")
        ax.set_title(title)
        ax.set_xlabel("Cosine distance")
        ax.set_ylabel("Count")
        ax.legend()
    fig.suptitle("Negative distance distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_distance_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── 5. Scene negative-usage balance (train) ──────────────────────────
    if train_sc:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Include scenes that were never used as a negative source (count = 0)
        # using the train_scenes set passed into the function
        all_counts = [train_sc.get(scene, 0) for scene in train_scenes]
        
        ax.hist(all_counts, bins=40, color="#55a868", edgecolor="white")
        
        mean_val = np.mean(all_counts)
        median_val = np.median(all_counts)
        max_val = np.max(all_counts)
        zero_scenes = sum(1 for v in all_counts if v == 0)
        
        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color="blue", linestyle="--", label=f"Median: {median_val:.0f}")
        
        # Add an explanatory text box with more context
        textstr = (f"Total Train Scenes: {len(all_counts)}\n"
                   f"Never used (0 negatives): {zero_scenes}\n"
                   f"Most used scene: {max_val} times")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.65, 0.50, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        ax.set_title("Negative Sampling Balance: Are negatives drawn from diverse scenes?")
        ax.set_xlabel("Number of negatives pulled from a single scene")
        ax.set_ylabel("Number of scenes")
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        fig.savefig(fig_dir / "05_scene_neg_balance.png", dpi=150)
        plt.close(fig)

    # ─── 6. Source breakdown of triplet roles ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (role, key) in zip(axes, [
        ("Anchor", "anchor_source"),
        ("Positive", "positive_source"),
        ("Negative", "negative_source"),
    ]):
        cc = Counter(t[key] for t in train_triplets)
        vals = [cc.get(s, 0) for s in sources]
        bars = ax.bar(sources, vals, color=["#4c72b0", "#55a868", "#c44e52"])
        ax.set_title(f"Train triplets: {role} source")
        ax.set_ylabel("Count")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 10, str(v),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "06_triplet_source_roles.png", dpi=150)
    plt.close(fig)

    # ─── 7. Per-source test ratio (actual vs target) ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, src in enumerate(sources):
        n_train = sum(1 for s in train_scenes if _scene_source(s) == src)
        n_test = sum(1 for s in test_scenes if _scene_source(s) == src)
        total = n_train + n_test
        ratio = n_test / total if total > 0 else 0
        ax.bar(i, ratio, color=["#4c72b0", "#55a868", "#c44e52"][i],
               label=f"{src} ({n_test}/{total})")
    ax.axhline(0.15, color="black", linestyle="--", label="Target 15%")
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources)
    ax.set_ylabel("Test ratio")
    ax.set_title("Actual test ratio per source")
    ax.set_ylim(0, 0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "07_per_source_test_ratio.png", dpi=150)
    plt.close(fig)

    print(f"  Figures saved to {fig_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def process_room(room_name: str, embeddings_path: Path, output_dir: Path):
    t0 = time.time()

    if not embeddings_path.exists():
        print(f"ERROR: embeddings file not found at {embeddings_path}")
        sys.exit(1)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 · Loading embeddings")
    print("=" * 60)
    all_items, scene_to_items, category_to_items = load_embeddings(embeddings_path)
    cats = sorted(category_to_items.keys())
    print(f"  Unique items: {len(all_items)}")
    print(f"  Scenes:       {len(scene_to_items)}")
    print(f"  Categories:   {cats}")
    for c in cats:
        print(f"    {c}: {len(category_to_items[c])} items")

    # Stats on multi-scene items
    multi = [it for it in all_items.values() if len(it["scenes"]) > 1]
    if multi:
        scene_counts = [len(it["scenes"]) for it in multi]
        print(f"  Multi-scene items: {len(multi)}"
              f"  (max {max(scene_counts)} scenes)")

    # ── 2. Split ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 · Train / test split")
    print("=" * 60)
    train_scenes, test_scenes = split_train_test(scene_to_items, TEST_RATIO, RANDOM_SEED)
    print(f"  Train scenes: {len(train_scenes)}")
    print(f"  Test  scenes: {len(test_scenes)}")

    # Per-source breakdown
    for src in set(_scene_source(s) for s in train_scenes | test_scenes):
        n_tr = sum(1 for s in train_scenes if _scene_source(s) == src)
        n_te = sum(1 for s in test_scenes if _scene_source(s) == src)
        total = n_tr + n_te
        ratio = n_te / total if total else 0
        print(f"    {src}: train={n_tr}, test={n_te}, ratio={ratio:.1%}")

    # Pair-eligible = scenes with ≥ 2 distinct categories
    multi_cat = lambda ss: {
        s for s in ss
        if len({it["category"] for it in scene_to_items[s]}) >= 2
    }
    train_pair_scenes = multi_cat(train_scenes)
    test_pair_scenes = multi_cat(test_scenes)
    print(f"  Train pair-eligible: {len(train_pair_scenes)}")
    print(f"  Test  pair-eligible: {len(test_pair_scenes)}")

    for label, ss in [("Train", train_scenes), ("Test", test_scenes)]:
        cc = Counter()
        for s in ss:
            for it in scene_to_items[s]:
                cc[it["category"]] += 1
        print(f"  {label} items by category: {dict(sorted(cc.items()))}")

    # ── 3. Pairs ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 · Generating anchor–positive pairs")
    print("=" * 60)
    train_pairs = generate_pairs(scene_to_items, train_pair_scenes)
    test_pairs = generate_pairs(scene_to_items, test_pair_scenes)
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Test  pairs: {len(test_pairs)}")

    pair_cats = Counter((a["category"], p["category"]) for a, p, _ in train_pairs)
    print("  Train pair categories:")
    for (ac, pc), cnt in sorted(pair_cats.items()):
        print(f"    ({ac}, {pc}): {cnt}")

    # ── 4. Negatives (train) ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 · Selecting negatives  [train]")
    print("=" * 60)
    train_triplets, train_skip, train_sc = select_negatives(
        train_pairs, scene_to_items, category_to_items, train_scenes,
        NUM_NEGATIVES_PER_PAIR, NEG_PERCENTILE_LOW, NEG_PERCENTILE_HIGH, RANDOM_SEED,
    )
    print(f"  Triplets: {len(train_triplets)}  |  skipped pairs: {train_skip}")
    if train_sc:
        vals = list(train_sc.values())
        print(
            f"  Scene neg usage — min: {min(vals)}, max: {max(vals)}, "
            f"mean: {np.mean(vals):.1f}, median: {np.median(vals):.0f}, "
            f"std: {np.std(vals):.1f}"
        )

    # ── 5. Negatives (test) ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 · Selecting negatives  [test]")
    print("=" * 60)
    test_triplets, test_skip, test_sc = select_negatives(
        test_pairs, scene_to_items, category_to_items, test_scenes,
        NUM_NEGATIVES_PER_PAIR, NEG_PERCENTILE_LOW, NEG_PERCENTILE_HIGH,
        RANDOM_SEED + 1,
    )
    print(f"  Triplets: {len(test_triplets)}  |  skipped pairs: {test_skip}")
    if test_sc:
        vals = list(test_sc.values())
        print(
            f"  Scene neg usage — min: {min(vals)}, max: {max(vals)}, "
            f"mean: {np.mean(vals):.1f}, median: {np.median(vals):.0f}, "
            f"std: {np.std(vals):.1f}"
        )

    # ── 6. Save ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 · Saving outputs")
    print("=" * 60)
    metadata = {
        "distance_metric": "cosine",
        "num_negatives_per_pair": NUM_NEGATIVES_PER_PAIR,
        "neg_percentile_range": [NEG_PERCENTILE_LOW, NEG_PERCENTILE_HIGH],
        "test_ratio": TEST_RATIO,
        "random_seed": RANDOM_SEED,
        "total_items": len(all_items),
        "total_scenes": len(scene_to_items),
        "categories": cats,
        "num_train_scenes": len(train_scenes),
        "num_test_scenes": len(test_scenes),
        "num_train_pair_scenes": len(train_pair_scenes),
        "num_test_pair_scenes": len(test_pair_scenes),
        "num_train_pairs": len(train_pairs),
        "num_test_pairs": len(test_pairs),
        "num_train_triplets": len(train_triplets),
        "num_test_triplets": len(test_triplets),
        "train_skipped_pairs": train_skip,
        "test_skipped_pairs": test_skip,
    }
    save_outputs(
        train_triplets, test_triplets,
        train_scenes, test_scenes,
        output_dir, metadata,
    )
    print(f"  Dir:   {output_dir}")
    print(f"  Files: triplets.json, train_triplets.csv, test_triplets.csv, scene_split.json")

    # ── 7. Embedding matrix + NPY export ──────────────────────────────────
    print("\n  Exporting compact embedding matrix …")
    shape = export_embedding_matrix(all_items, output_dir)
    print(f"  embeddings.npz  shape={shape}")
    print(f"  embedding_index.json")

    print("\n  Exporting triplet index arrays (.npy) …")
    tr_shape = export_triplet_npy(train_triplets, all_items, output_dir / "train_triplet_indices.npy")
    te_shape = export_triplet_npy(test_triplets, all_items, output_dir / "test_triplet_indices.npy")
    print(f"  train_triplet_indices.npy  shape={tr_shape}")
    print(f"  test_triplet_indices.npy   shape={te_shape}")

    # ── 8. Visualizations ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8 · Generating visualizations")
    print("=" * 60)
    plot_visualizations(
        train_triplets, test_triplets,
        train_scenes, test_scenes,
        scene_to_items, train_sc, test_sc,
        output_dir,
    )

    elapsed = time.time() - t0
    print(f"\nDone  ({elapsed:.1f} s)")


def main():
    for room in ROOMS_TO_PROCESS:
        print(f"Processing Room: {room}")
        embeddings_path = BASE_DIR / "data" / "ml_data" / room / "embeddings" / "furniture_embeddings.json"
        output_dir = BASE_DIR / "data" / "ml_data" / room / "triplets"
        if not embeddings_path.exists():
            print(f"Skipping {room}: No embeddings found at {embeddings_path}")
            continue
        process_room(room, embeddings_path, output_dir)

if __name__ == "__main__":
    main()


