import argparse
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--room", default="bedrooms", choices=["bedrooms", "living_rooms"])
args = parser.parse_args()

NPZ_PATH = BASE_DIR / "data" / "retrieval_data" / args.room / "retrieval_embeddings.npz"

print(f"Room: {args.room}")
print(f"Path: {NPZ_PATH}")

data = np.load(NPZ_PATH, allow_pickle=False)

embeddings = data["embeddings"]   # (N, 128) float32
ids        = data["ids"]          # (N,)

print("Keys:            ", list(data.keys()))
print("Embeddings shape:", embeddings.shape)
print("Embeddings dtype:", embeddings.dtype)
print("IDs shape:       ", ids.shape)
print("First 5 IDs:     ", ids[:5])
print("First embedding: ", embeddings[0, :8], "...")

norms = np.linalg.norm(embeddings, axis=1)
print(f"Norm min/max:    {norms.min():.6f} / {norms.max():.6f}  (should be ~1.0)")