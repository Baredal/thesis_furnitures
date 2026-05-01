import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))
from ml.model import SiameseResnet18

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed_data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOMS_TO_PROCESS = ["bedrooms", "living_rooms"]

CHECKPOINT_NAMES = {
    "bedrooms":     "best_model_bedrooms.pt",
    "living_rooms": "best_model_liv_rooms.pt",
}

def build_href_map(room) -> dict[str, str]:
    href_map: dict[str, str] = {}
    room_dir = PROCESSED_DATA_DIR / room
    if not room_dir.exists():
        return href_map
    for ann_path in room_dir.rglob("*annotations.json"):
        try:
            ann = json.load(open(ann_path, encoding="utf-8"))
        except Exception:
            continue
        for furn in ann.get("furnitures", []):
            fid = str(furn.get("furniture_id", ""))
            href = furn.get("furniture_href")
            if fid and href and fid not in href_map:
                href_map[fid] = href
    return href_map

def process_room(room):
    manifest_path = BASE_DIR / "data" / "total" / room / "general_manifest.json"
    image_root    = BASE_DIR / "data" / "total" / room
    model_path    = BASE_DIR / "data" / "ml_data" / room / "models" / CHECKPOINT_NAMES[room]
    output_dir    = BASE_DIR / "data" / "retrieval_data" / room

    if not manifest_path.exists():
        print(f"[{room}] Manifest not found, skipping."); return
    if not model_path.exists():
        print(f"[{room}] Checkpoint not found: {model_path}, skipping."); return

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path   = output_dir / "retrieval_embeddings.npz"
    index_path = output_dir / "retrieval_index.json"

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)
    cfg = ckpt["config"]
    model = SiameseResnet18(embedding_dim=cfg["embedding_dim"], pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    furnitures = json.load(open(manifest_path, encoding="utf-8"))["furnitures"]
    href_map   = build_href_map(room)

    all_embeddings = []
    all_ids = []
    index = {}
    row = 0

    with torch.no_grad():
        for item in tqdm(furnitures, desc=f"Embedding {room}"):
            fid      = item["furniture_id"]
            category = item["category"]
            img_path = image_root / category / item["image_name"]
            if not img_path.exists():
                continue

            img    = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            emb    = model(tensor).squeeze(0).cpu().numpy()

            all_embeddings.append(emb)
            all_ids.append(fid)
            index[str(row)] = {
                "furniture_id": fid,
                "category":     category,
                "source":       item["source"],
                "scene":        item["scene"],
                "image_name":   item["image_name"],
                "furniture_href": href_map.get(str(fid)),
            }
            row += 1

    if not all_embeddings:
        print(f"[{room}] No embeddings produced, skipping save."); return

    np.savez_compressed(npz_path, embeddings=np.array(all_embeddings, dtype=np.float32), ids=np.array(all_ids))
    json.dump(index, open(index_path, "w", encoding="utf-8"), indent=2)
    print(f"[{room}] Saved {row} embeddings → {output_dir}")

if __name__ == "__main__":
    for room in ROOMS_TO_PROCESS: process_room(room)
