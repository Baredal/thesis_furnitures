import os
import shutil
import subprocess
from pathlib import Path

HF_USERNAME  = "Darebal"
MODEL_REPO   = f"https://huggingface.co/{HF_USERNAME}/furniture-compatibility-siamese"
DATASET_REPO = f"https://huggingface.co/datasets/{HF_USERNAME}/furniture-catalog"

BASE_DIR = Path(__file__).resolve().parent
STAGING_DIR = BASE_DIR / "hf_staging"

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

def clone_repos():
    """Clones the repos via Git LFS into a staging directory."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Clone Model Repo
    model_dir = STAGING_DIR / "models_staging"
    if not model_dir.exists():
        print(f"\nCloning Model Repository: {MODEL_REPO}...")
        subprocess.run(["git", "clone", MODEL_REPO, str(model_dir)], check=True)
    else:
        print(f"\nModel staging directory already exists. Skipping clone.")
    
    # 2. Clone Dataset Repo
    dataset_dir = STAGING_DIR / "dataset_staging"
    if not dataset_dir.exists():
        print(f"\nCloning Dataset Repository: {DATASET_REPO}...")
        subprocess.run(["git", "clone", DATASET_REPO, str(dataset_dir)], check=True)
    else:
        print(f"\nDataset staging directory already exists. Skipping clone.")

    return model_dir, dataset_dir

def distribute_models(model_dir):
    print("\nMoving model artefacts...")
    for repo_path, local_path in MODEL_FILES.items():
        src = model_dir / repo_path
        if src.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(local_path))
            print(f"  → Moved to {local_path.name}")

def distribute_training_data(dataset_dir):
    print("\nMoving training data...")
    for repo_path, local_path in TRAINING_FILES.items():
        src = dataset_dir / repo_path
        if src.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(local_path))
            print(f"  → Moved to {local_path.name}")

def distribute_scenes(dataset_dir):
    print("\nMoving scene images...")
    src = dataset_dir / "scenes"
    dest = BASE_DIR / "data" / "processed_data"
    if src.exists():
        dest.mkdir(parents=True, exist_ok=True)
        # Move all contents of scenes/ into processed_data/
        for item in src.iterdir():
            shutil.move(str(item), str(dest / item.name))
        print(f"  → Scene images moved to {dest}")

def distribute_catalog(dataset_dir):
    print("\nMoving furniture catalog images...")
    dest = BASE_DIR / "data" / "total"
    dest.mkdir(parents=True, exist_ok=True)
    
    # Files and folders to completely ignore
    ignore_names = {".git", ".gitattributes", "README.md", "scenes", "triplets", "embeddings"}
    
    # os.walk goes through every folder and sub-folder perfectly
    for root, dirs, files in os.walk(dataset_dir):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_names]
        
        for file in files:
            if file in ignore_names:
                continue
                
            # Calculate where the file should go based on its subfolder structure
            src_path = Path(root) / file
            rel_path = src_path.relative_to(dataset_dir)
            dest_path = dest / rel_path
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dest_path))
            
    print(f"  → Catalog moved to {dest}")

def cleanup():
    print("\nCleaning up staging directory...")
    if STAGING_DIR.exists():
        # Handle read-only files (.git folder issues on Windows)
        def handle_remove_readonly(func, path, exc):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(STAGING_DIR, onerror=handle_remove_readonly)
        print("  → Staging directory removed.")

if __name__ == "__main__":
    print("Starting Git LFS bulk download process...")
    # Make sure we don't need a token variable anymore
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]

    model_dir, dataset_dir = clone_repos()
    
    distribute_models(model_dir)
    distribute_training_data(dataset_dir)
    distribute_scenes(dataset_dir)
    distribute_catalog(dataset_dir)
    
    cleanup()
    
    print("\nAll done. Run: streamlit run src/app/streamlit_app.py")