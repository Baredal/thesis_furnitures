from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
RETRIEVAL_DIR = BASE_DIR / "data" / "retrieval_data"
IMAGE_ROOT = BASE_DIR / "data" / "total"

def compute_from_pil(pil_image: Image.Image, bins: int = 32) -> np.ndarray:
    """Same encoding as compute() but accepts a PIL image directly."""
    img = pil_image.convert("RGB").resize((64, 64))
    arr = np.array(img)
    hist = np.concatenate([
        np.histogram(arr[:, :, c], bins=bins, range=(0, 256))[0]
        for c in range(3)
    ]).astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return np.sqrt(hist).astype(np.float32)


def compute(image_path: Path, bins: int = 32) -> np.ndarray:
    """
    RGB histogram encoded for Bhattacharyya via dot product.
    Stores sqrt(p) so that:  dot(a, b) == BC(p_a, p_b) ∈ [0, 1]
    """
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    arr = np.array(img)
    hist = np.concatenate([
        np.histogram(arr[:, :, c], bins=bins, range=(0, 256))[0]
        for c in range(3)
    ]).astype(np.float64)

    total = hist.sum()
    if total > 0:
        hist /= total
    return np.sqrt(hist).astype(np.float32)


def load_or_build(hists_path: Path, index: dict[int, dict], image_root: Path = IMAGE_ROOT) -> np.ndarray:
    N = len(index)
    if hists_path.exists():
        cached = np.load(hists_path, allow_pickle=False)["histograms"]
        if cached.shape[0] == N:
            print("Loading histogram cache...")
            return cached
        print(f"Histogram cache size mismatch ({cached.shape[0]} vs {N} items), rebuilding...")

    print("Computing Bhattacharyya histograms (one-time, will be cached)...")
    N     = len(index)
    hists = np.zeros((N, 96), dtype=np.float32)   # 3 channels × 32 bins

    for row, meta in tqdm(index.items(), total=N, desc="Histograms"):
        img_path = image_root / meta["category"] / meta["image_name"]
        if img_path.exists():
            hists[row] = compute(img_path)

    np.savez_compressed(hists_path, histograms=hists)
    print(f"Saved → {hists_path}")
    return hists
