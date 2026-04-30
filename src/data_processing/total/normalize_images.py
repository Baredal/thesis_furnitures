"""
Normalize furniture images in data/total/ to 512×512 JPEG.

Resizes with LANCZOS, pads to a square white canvas, and saves
at quality=95. Run once after (re)building data/total/.

Usage:
    python src/data_processing/normalize_images.py               # all rooms
    python src/data_processing/normalize_images.py --room bedrooms
    python src/data_processing/normalize_images.py --dry-run
"""

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[3]
TOTAL_DIR = BASE_DIR / "data" / "total"

TARGET  = 1024
QUALITY = 95


def normalize(img_path: Path) -> None:
    img = Image.open(img_path).convert("RGB")
    # Scale so the longest side == TARGET (upscale AND downscale)
    scale = TARGET / max(img.width, img.height)
    new_w = round(img.width * scale)
    new_h = round(img.height * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (TARGET, TARGET), (255, 255, 255))
    offset = ((TARGET - new_w) // 2, (TARGET - new_h) // 2)
    canvas.paste(img, offset)
    canvas.save(img_path, "JPEG", quality=QUALITY, optimize=True)


def run_room(room: str, dry_run: bool) -> None:
    room_dir = TOTAL_DIR / room
    if not room_dir.exists():
        print(f"  [skip] {room_dir} does not exist")
        return

    images = list(room_dir.rglob("*.jpg")) + list(room_dir.rglob("*.jpeg"))
    print(f"\n{room}: {len(images)} images found")

    if dry_run:
        print("  [dry-run] no changes written")
        return

    for path in tqdm(images, desc=room):
        try:
            normalize(path)
        except Exception as exc:
            print(f"  [warn] {path.name}: {exc}")

    print(f"  Done — {len(images)} images normalized to {TARGET}×{TARGET}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--room", choices=["bedrooms", "living_rooms"],
                        help="Process one room only (default: both)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count files without writing anything")
    args = parser.parse_args()

    rooms = [args.room] if args.room else ["bedrooms", "living_rooms"]
    for room in rooms:
        run_room(room, dry_run=args.dry_run)
