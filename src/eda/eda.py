import argparse
import csv
import json
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]

ROOM_CATEGORIES = {
    "bedrooms":    ["bed", "small_storage", "large_storage", "table", "chair_stool", "curtain"],
    "living_rooms": ["sofa", "small_storage", "large_storage", "table", "chair_stool", "curtain"],
}
ROOM_SOURCES = {
    "bedrooms":    ["deepfurn", "sklad_mebliv"],
    "living_rooms": ["deepfurn"],
}

COLORS = {"deepfurn": "#4c72b0", "sklad_mebliv": "#c44e52"}
SPLIT_COLORS = {"train": "#4c72b0", "val": "#55a868", "test": "#c44e52"}


@dataclass
class RoomCtx:
    room:          str
    categories:    list
    sources:       list
    total_dir:     Path
    processed_dir: Path
    triplet_dir:   Path
    output_dir:    Path

    def manifest(self) -> list[dict]:
        with open(self.total_dir / "general_manifest.json", encoding="utf-8") as f:
            return json.load(f)["furnitures"]

    def triplets(self, split: str) -> list[dict]:
        with open(self.triplet_dir / f"{split}_triplets.csv", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def scene_split(self) -> dict:
        with open(self.triplet_dir / "scene_split.json", encoding="utf-8") as f:
            return json.load(f)


def make_ctx(room: str) -> RoomCtx:
    output_dir = BASE_DIR / "data" / "eda_output" / room
    output_dir.mkdir(parents=True, exist_ok=True)
    return RoomCtx(
        room          = room,
        categories    = ROOM_CATEGORIES[room],
        sources       = ROOM_SOURCES[room],
        total_dir     = BASE_DIR / "data" / "total" / room,
        processed_dir = BASE_DIR / "data" / "processed_data" / room,
        triplet_dir   = BASE_DIR / "data" / "ml_data" / room / "triplets_v3",
        output_dir    = output_dir,
    )


def scene_source(name: str) -> str:
    for prefix in ("deepfurn", "sklad_mebliv", "wayfair"):
        if name.startswith(prefix):
            return prefix
    return "unknown"


def print_table(title: str, headers: list[str], rows: list[list], file=None):
    col_widths = [max(len(str(cell)) for cell in [h] + [r[i] for r in rows])
                  for i, h in enumerate(headers)]

    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, col_widths)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    lines = [f"\n{title}\n", fmt_row(headers), sep] + [fmt_row(r) for r in rows]
    text = "\n".join(lines)
    print(text)
    if file:
        file.write(text + "\n\n")


def build_tables(ctx: RoomCtx):
    report = open(ctx.output_dir / "tables.txt", "w", encoding="utf-8")
    items = ctx.manifest()
    split = ctx.scene_split()

    counts = Counter((it["source"], it["category"]) for it in items)
    headers = ["Category"] + ctx.sources + ["Total"]
    rows = []
    col_totals = {s: 0 for s in ctx.sources}
    for cat in ctx.categories:
        row = [cat]
        cat_total = 0
        for s in ctx.sources:
            v = counts.get((s, cat), 0)
            row.append(str(v) if v else "-")
            cat_total += v
            col_totals[s] += v
        row.append(str(cat_total))
        rows.append(row)
    rows.append(["Total"] + [str(col_totals[s]) for s in ctx.sources] +
                [str(sum(col_totals.values()))])
    print_table(f"Table 1 — Furniture items by category and source ({ctx.room})", headers, rows, report)

    headers = ["Split", "Scenes"] + ctx.sources
    rows = []
    for split_name in ["train_scenes", "val_scenes", "golden_scenes"]:
        scenes = split[split_name]
        src_c = Counter(scene_source(s) for s in scenes)
        label = split_name.replace("_scenes", "").replace("golden", "test")
        rows.append([label, str(len(scenes))] + [str(src_c.get(s, 0)) for s in ctx.sources])
    total_scenes = sum(len(split[k]) for k in split)
    rows.append(["Total", str(total_scenes)] + [""] * len(ctx.sources))
    print_table(f"Table 2 — Three-way scene split ({ctx.room})", headers, rows, report)

    headers = ["Split", "Triplets", "Unique items", "Unique pairs"]
    rows = []
    for split_name in ["train", "val", "golden"]:
        triplets = ctx.triplets(split_name)
        item_ids = set()
        pairs = set()
        for r in triplets:
            item_ids.update([r["anchor_id"], r["positive_id"], r["negative_id"]])
            pairs.add((r["anchor_id"], r["positive_id"]))
        label = "test" if split_name == "golden" else split_name
        rows.append([label, str(len(triplets)), str(len(item_ids)), str(len(pairs))])
    print_table(f"Table 3 — Triplet dataset statistics ({ctx.room})", headers, rows, report)

    all_training_scenes = split["train_scenes"] + split["val_scenes"] + split["golden_scenes"]
    split_scene_sets = defaultdict(set)
    for s in all_training_scenes:
        split_scene_sets[scene_source(s)].add(s)

    item_counts = Counter(it["source"] for it in items)
    headers = ["Source", "Scenes (split)", "Items (manifest)", "Items/scene"]
    rows = []
    for s in ctx.sources:
        n_sc = len(split_scene_sets[s])
        n_it = item_counts[s]
        rows.append([s, str(n_sc), str(n_it), f"{n_it / n_sc:.1f}" if n_sc else "-"])
    print_table(f"Table 4 — Source-level statistics ({ctx.room})", headers, rows, report)

    report.close()
    print(f"  Tables saved → {ctx.output_dir / 'tables.txt'}")


def fig1_category_by_source(ctx: RoomCtx):
    items = ctx.manifest()
    counts = Counter((it["source"], it["category"]) for it in items)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ctx.categories))
    width = 0.8 / len(ctx.sources)
    for i, src in enumerate(ctx.sources):
        vals = [counts.get((src, c), 0) for c in ctx.categories]
        bars = ax.bar(x + i * width, vals, width, label=src, color=COLORS.get(src, "#888"))
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 5,
                        str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x + width * (len(ctx.sources) - 1) / 2)
    ax.set_xticklabels(ctx.categories, rotation=20, ha="right")
    ax.set_ylabel("Items")
    ax.set_title(f"Items per category by source ({ctx.room})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ctx.output_dir / "01_category_by_source.png", dpi=150)
    plt.close(fig)


def fig2_items_per_scene(ctx: RoomCtx):
    items = ctx.manifest()
    scene_sizes = Counter(it["scene"] for it in items)
    size_dist = Counter(scene_sizes.values())
    sizes = sorted(size_dist)
    freqs = [size_dist[s] for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(sizes, freqs, color="#4c72b0", edgecolor="white")
    for s, f in zip(sizes, freqs):
        ax.text(s, f + 1, str(f), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Items per scene")
    ax.set_ylabel("Number of scenes")
    ax.set_title(f"Items per scene distribution ({ctx.room}, {len(scene_sizes)} scenes)")
    ax.set_xticks(sizes)
    fig.tight_layout()
    fig.savefig(ctx.output_dir / "02_items_per_scene.png", dpi=150)
    plt.close(fig)


def fig3_scene_split_by_source(ctx: RoomCtx):
    split = ctx.scene_split()
    split_names = [("train", "train"), ("val", "val"), ("golden", "test")]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ctx.sources))
    width = 0.25
    for i, (file_name, label) in enumerate(split_names):
        scenes = split[f"{file_name}_scenes"]
        vals = [sum(1 for s in scenes if scene_source(s) == src) for src in ctx.sources]
        bars = ax.bar(x + i * width, vals, width, label=label, color=SPLIT_COLORS[label])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                        str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x + width)
    ax.set_xticklabels(ctx.sources)
    ax.set_ylabel("Scenes")
    ax.set_title(f"Three-way scene split by source ({ctx.room})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ctx.output_dir / "03_scene_split_by_source.png", dpi=150)
    plt.close(fig)


def fig4_category_pair_heatmap(ctx: RoomCtx):
    triplets = ctx.triplets("train")
    pair_counts = Counter()
    for r in triplets:
        a, p = sorted([r["anchor_category"], r["positive_category"]])
        pair_counts[(a, p)] += 1

    cats = ctx.categories
    matrix = np.zeros((len(cats), len(cats)), dtype=int)
    for (a, p), cnt in pair_counts.items():
        if a in cats and p in cats:
            i, j = cats.index(a), cats.index(p)
            matrix[i][j] = cnt
            matrix[j][i] = cnt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(cats)))
    ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=35, ha="right")
    ax.set_yticklabels(cats)
    for i in range(len(cats)):
        for j in range(len(cats)):
            v = matrix[i][j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if v > matrix.max() * 0.6 else "black")
    ax.set_title(f"Category pair counts in train triplets ({ctx.room})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(ctx.output_dir / "04_category_pair_heatmap.png", dpi=150)
    plt.close(fig)


def fig5_neg_distance_distribution(ctx: RoomCtx):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for file_name, label in [("train", "train"), ("val", "val"), ("golden", "test")]:
        triplets = ctx.triplets(file_name)
        dists = [float(r["combined_distance"]) for r in triplets]
        ax.hist(dists, bins=60, alpha=0.5, label=f"{label} (n={len(dists)})",
                color=SPLIT_COLORS[label])
    ax.set_xlabel("Combined distance (pos-neg + anchor-counterpart)")
    ax.set_ylabel("Count")
    ax.set_title(f"Negative selection distance distribution ({ctx.room})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ctx.output_dir / "05_neg_distance_distribution.png", dpi=150)
    plt.close(fig)


def fig6_sample_scenes(ctx: RoomCtx, n_samples=3, seed=42):
    rng = random.Random(seed)
    items = ctx.manifest()
    scene_items = defaultdict(list)
    for it in items:
        scene_items[it["scene"]].append(it)

    good_scenes = [s for s, its in scene_items.items() if len(its) >= 4]
    chosen = rng.sample(good_scenes, min(n_samples, len(good_scenes)))

    for scene_name in chosen:
        source = scene_source(scene_name)
        scene_dir = ctx.processed_dir / source / scene_name
        scene_img_path = scene_dir / "scene_image.jpg"
        furnitures_dir = scene_dir / "furnitures"

        if not scene_img_path.exists():
            continue

        furn_items = scene_items[scene_name]
        n_furn = len(furn_items)

        fig, axes = plt.subplots(1, 1 + n_furn, figsize=(3 * (1 + n_furn), 3.5))

        axes[0].imshow(Image.open(scene_img_path))
        axes[0].set_title("Scene", fontsize=10)
        axes[0].axis("off")

        for j, it in enumerate(furn_items):
            fpath = furnitures_dir / f"{it['furniture_id']}.jpg"
            if fpath.exists():
                axes[j + 1].imshow(Image.open(fpath))
            else:
                axes[j + 1].text(0.5, 0.5, "?", ha="center", va="center", fontsize=20)
            axes[j + 1].set_title(it["category"], fontsize=9)
            axes[j + 1].axis("off")

        fig.suptitle(f"{scene_name}  ({source})", fontsize=11)
        fig.tight_layout()
        fig.savefig(ctx.output_dir / f"sample_{scene_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_room(room: str):
    print(f"\n{'=' * 60}")
    print(f"  EDA — {room}")
    print(f"{'=' * 60}")
    ctx = make_ctx(room)
    build_tables(ctx)
    fig1_category_by_source(ctx)
    fig2_items_per_scene(ctx)
    fig3_scene_split_by_source(ctx)
    fig4_category_pair_heatmap(ctx)
    fig5_neg_distance_distribution(ctx)
    fig6_sample_scenes(ctx)
    print(f"  All outputs saved → {ctx.output_dir}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--room", choices=["bedrooms", "living_rooms"],
                        help="Run EDA for a single room (default: both)")
    args = parser.parse_args()

    rooms = [args.room] if args.room else ["bedrooms", "living_rooms"]
    for room in rooms:
        run_room(room)
