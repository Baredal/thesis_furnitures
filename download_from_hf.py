import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

HF_USERNAME  = "Darebal"
MODEL_REPO   = f"{HF_USERNAME}/furniture-compatibility-siamese"
DATASET_REPO = f"{HF_USERNAME}/furniture-catalog"

BASE_DIR = Path(__file__).resolve().parent
TOKEN    = os.getenv("HF_TOKEN")  # optional — repos are public

MODEL_FILES = {
    "bedrooms/best_model_bedrooms.pt":
        BASE_DIR / "data/ml_data/bedrooms/models/best_model_bedrooms.pt",
    "bedrooms/retrieval_embeddings.npz":
        BASE_DIR / "data/retrieval_data/bedrooms/retrieval_embeddings.npz",
    "bedrooms/retrieval_index.json":
        BASE_DIR / "data/retrieval_data/bedrooms/retrieval_index.json",
    "bedrooms/retrieval_histograms_bc.npz":
        BASE_DIR / "data/retrieval_data/bedrooms/retrieval_histograms_bc.npz",

    "living_rooms/best_model_liv_rooms.pt":
        BASE_DIR / "data/ml_data/living_rooms/models/best_model_liv_rooms.pt",
    "living_rooms/retrieval_embeddings.npz":
        BASE_DIR / "data/retrieval_data/living_rooms/retrieval_embeddings.npz",
    "living_rooms/retrieval_index.json":
        BASE_DIR / "data/retrieval_data/living_rooms/retrieval_index.json",
    "living_rooms/retrieval_histograms_bc.npz":
        BASE_DIR / "data/retrieval_data/living_rooms/retrieval_histograms_bc.npz",
}

TRAINING_FILES = {
    "bedrooms/triplets/train_triplets.csv":
        BASE_DIR / "data/ml_data/bedrooms/triplets_v3/train_triplets.csv",
    "bedrooms/triplets/val_triplets.csv":
        BASE_DIR / "data/ml_data/bedrooms/triplets_v3/val_triplets.csv",
    "bedrooms/triplets/golden_triplets.csv":
        BASE_DIR / "data/ml_data/bedrooms/triplets_v3/golden_triplets.csv",
    "bedrooms/triplets/scene_split.json":
        BASE_DIR / "data/ml_data/bedrooms/triplets_v3/scene_split.json",
    "living_rooms/triplets/train_triplets.csv":
        BASE_DIR / "data/ml_data/living_rooms/triplets_v3/train_triplets.csv",
    "living_rooms/triplets/val_triplets.csv":
        BASE_DIR / "data/ml_data/living_rooms/triplets_v3/val_triplets.csv",
    "living_rooms/triplets/golden_triplets.csv":
        BASE_DIR / "data/ml_data/living_rooms/triplets_v3/golden_triplets.csv",
    "living_rooms/triplets/scene_split.json":
        BASE_DIR / "data/ml_data/living_rooms/triplets_v3/scene_split.json",

    "bedrooms/embeddings/furniture_embeddings.json":
        BASE_DIR / "data/ml_data/bedrooms/embeddings/furniture_embeddings.json",
    "living_rooms/embeddings/furniture_embeddings.json":
        BASE_DIR / "data/ml_data/living_rooms/embeddings/furniture_embeddings.json",
}


def download_models():
    print(f"Downloading model artefacts from {MODEL_REPO}...")
    missing = []
    for repo_path, local_path in MODEL_FILES.items():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            print(f"  already exists — skipping: {local_path.name}")
            continue
        print(f"  {repo_path}...")
        try:
            tmp = hf_hub_download(repo_id=MODEL_REPO, filename=repo_path,
                                  repo_type="model", token=TOKEN)
            shutil.copy(tmp, local_path)
            print(f"    → {local_path}")
        except EntryNotFoundError:
            print(f"  WARNING: not found in repo — {repo_path}")
            missing.append(repo_path)
    if missing:
        print("\n  The following files are not yet uploaded to the model repo:")
        for p in missing:
            print(f"    {MODEL_REPO}/{p}")
        print("  Upload them with: huggingface-cli upload or the HuggingFace web UI.")


def download_catalog():
    print(f"\nDownloading furniture catalog images from {DATASET_REPO}...")
    dest = BASE_DIR / "data" / "total"
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(dest),
        token=TOKEN,
        ignore_patterns=[
            "*.gitattributes", ".gitattributes", "README.md",
            "*/triplets/*", "*/embeddings/*", "scenes/**",
        ],
    )
    print(f"  Catalog saved to {dest}")


def download_training_data():
    print(f"\nDownloading training data from {DATASET_REPO}...")
    for repo_path, local_path in TRAINING_FILES.items():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            print(f"  already exists — skipping: {local_path.name}")
            continue
        print(f"  {repo_path}...")
        tmp = hf_hub_download(repo_id=DATASET_REPO, filename=repo_path,
                              repo_type="dataset", token=TOKEN)
        shutil.copy(tmp, local_path)
        print(f"    → {local_path}")


def download_scenes():
    print(f"\nDownloading scene images from {DATASET_REPO}...")
    dest = BASE_DIR / "data" / "processed_data"
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            local_dir=tmp,
            token=TOKEN,
            allow_patterns=["scenes/**"],
            ignore_patterns=["*.gitattributes", ".gitattributes"],
        )
        scenes_root = Path(tmp) / "scenes"
        if scenes_root.exists():
            shutil.copytree(str(scenes_root), str(dest), dirs_exist_ok=True)
    print(f"  Scene images saved to {dest}")


if __name__ == "__main__":
    download_models()
    download_catalog()
    download_training_data()
    download_scenes()
    print("\nAll done. Run: streamlit run src/app/streamlit_app.py")
