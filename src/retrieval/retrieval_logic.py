import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional
import sys
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src" / "retrieval"))
import histogram_filter

CATEGORY_CHAINS = {
    "bedrooms":     ["bed", "small_storage", "large_storage", "table", "chair_stool", "curtain"],
    "living_rooms": ["sofa", "table", "small_storage", "large_storage", "chair_stool", "curtain"],
}
CATEGORY_CHAIN = CATEGORY_CHAINS["bedrooms"]  # backwards-compat alias

class FurnitureRetriever:
    """
    Score = embed_weight * cosine(embedding)  +  hist_weight * Bhattacharyya(histogram)

    Returns plain dicts:
        furniture_id, category, source, scene, image_name, image_path,
        score, embed_score, hist_score
    """

    def __init__(self, room="bedrooms", embed_weight: float = 0.5, hist_weight: float = 0.5):
        self.room = room
        self.retrieval_dir = BASE_DIR / "data" / "retrieval_data" / room
        self.image_root = BASE_DIR / "data" / "total" / room

        npz_path   = self.retrieval_dir / "retrieval_embeddings.npz"
        index_path = self.retrieval_dir / "retrieval_index.json"
        hists_path = self.retrieval_dir / "retrieval_histograms_bc.npz"

        data = np.load(npz_path, allow_pickle=False)
        self._embeddings: np.ndarray = data["embeddings"]
        with open(index_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._index: dict[int, dict] = {int(k): v for k, v in raw.items()}
        self._histograms = histogram_filter.load_or_build(hists_path, self._index, self.image_root)
        self.embed_weight = embed_weight
        self.hist_weight = hist_weight

        self._category_rows: dict[str, list[int]] = defaultdict(list)
        self._id_to_row: dict[str, int] = {}
        for row, meta in self._index.items():
            self._category_rows[meta["category"]].append(row)
            self._id_to_row[meta["furniture_id"]] = row

        self._external: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._model = None
        self._transform = None
        self._device = None

        N = self._embeddings.shape[0]
        print(f"FurnitureRetriever ready — {N} items, {len(self._category_rows)} categories")
        for cat in sorted(self._category_rows):
            print(f"  {cat:15s}: {len(self._category_rows[cat])} items")

    _CKPT_NAMES = {
        "bedrooms":     "v3/best_model_v3_resnet18_new_data.pt",
        "living_rooms": "v3/best_model_v3_resnet18_liv_rooms.pt",
    }

    def _load_embed_model(self):
        if self._model is not None:
            return
        import torch
        from torchvision import transforms
        from ml.model import SiameseResnet18
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = BASE_DIR / "data" / "ml_data" / self.room / "models" / self._CKPT_NAMES[self.room]
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        cfg = ckpt["config"]
        model = SiameseResnet18(embedding_dim=cfg["embedding_dim"], pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self._model = model
        self._device = device
        self._transform = transforms.Compose([
            transforms.Resize((cfg["img_size"], cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def embed_image(self, pil_image) -> tuple[np.ndarray, np.ndarray]:
        """Embed a PIL image; returns (embedding, histogram) ready for get_compatible."""
        import torch
        self._load_embed_model()
        tensor = self._transform(pil_image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model(tensor).squeeze(0).cpu().numpy()
        hist = histogram_filter.compute_from_pil(pil_image)
        return emb, hist

    def register_external(self, fid: str, embedding: np.ndarray, histogram: np.ndarray):
        self._external[fid] = (embedding, histogram)

    def clear_external(self, fid: str):
        self._external.pop(fid, None)

    def categories(self) -> list[str]:
        return list(self._category_rows.keys())

    def get_random(self, category: str, n: int = 5) -> List[dict]:
        rows   = self._category_rows[category]
        chosen = random.sample(rows, min(n, len(rows)))
        return [self._to_dict(r) for r in chosen]

    def get_compatible(
        self,
        selected:        List[dict],
        target_category: str,
        top_k:           int = 5,
        weights:         Optional[List[float]] = None,
        exclude_ids:     Optional[set] = None,
    ) -> List[dict]:
        if not selected:
            return self.get_random(target_category, n=top_k)

        n_sel = len(selected)
        if weights is None:
            weights = (
                [0.3 + 0.7 * (i / (n_sel - 1)) for i in range(n_sel)]
                if n_sel > 1 else [1.0]
            )
        total_w = sum(weights)

        candidate_rows = self._category_rows.get(target_category, [])
        if exclude_ids:
            candidate_rows = [
                r for r in candidate_rows
                if self._index[r]["furniture_id"] not in exclude_ids
            ]
        if not candidate_rows:
            return []

        cand_arr   = np.array(candidate_rows)
        cand_embs  = self._embeddings[cand_arr]
        cand_hists = self._histograms[cand_arr]

        embed_scores = np.zeros(len(candidate_rows), dtype=np.float32)
        hist_scores  = np.zeros(len(candidate_rows), dtype=np.float32)

        for item, w in zip(selected, weights):
            row = self._id_to_row.get(item["furniture_id"])
            if row is not None:
                embed_scores += w * (cand_embs  @ self._embeddings[row])
                hist_scores  += w * (cand_hists @ self._histograms[row])
            else:
                ext = self._external.get(item["furniture_id"])
                if ext is not None:
                    embed_scores += w * (cand_embs  @ ext[0])
                    hist_scores  += w * (cand_hists @ ext[1])

        embed_scores /= total_w
        hist_scores  /= total_w

        final_scores = self.embed_weight * embed_scores + self.hist_weight * hist_scores

        k         = min(top_k, len(final_scores))
        top_local = np.argpartition(final_scores, -k)[-k:]
        top_local = top_local[np.argsort(final_scores[top_local])[::-1]]

        return [
            self._to_dict(
                candidate_rows[i],
                score       = float(final_scores[i]),
                embed_score = float(embed_scores[i]),
                hist_score  = float(hist_scores[i]),
            )
            for i in top_local
        ]

    def _to_dict(self, row: int, score=0.0, embed_score=0.0, hist_score=0.0) -> dict:
        meta = self._index[row]
        return {
            "furniture_id": meta["furniture_id"],
            "category":     meta["category"],
            "source":       meta["source"],
            "scene":        meta["scene"],
            "image_name":   meta["image_name"],
            "image_path":   str(self.image_root / meta["category"] / meta["image_name"]),
            "furniture_href": meta.get("furniture_href"),
            "score":        round(score, 4),
            "embed_score":  round(embed_score, 4),
            "hist_score":   round(hist_score, 4),
        }
