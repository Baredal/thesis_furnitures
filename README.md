# Visual Furnishings Compatibility Search Using Machine Learning

Bachelor's thesis · Ukrainian Catholic University · 2026

**HuggingFace:** [Model weights](https://huggingface.co/Darebal/furniture-compatibility-siamese) · [Dataset](https://huggingface.co/datasets/Darebal/furniture-catalog)

---

## About

This repository is the full code base accompanying the bachelor's thesis of the same name. The thesis covers dataset construction, triplet-based metric learning, a hybrid retrieval system, and an interactive room-building demo. The primary contribution is the code and trained models; the written thesis document lives separately and is referenced in the citation below.

The project addresses visual furniture compatibility: given a selected furniture item, retrieve other items that are stylistically compatible with it. A Siamese ResNet18 is trained with triplet margin loss on a multi-source dataset covering two room types (bedrooms, living rooms). A Streamlit app lets a user build a room step-by-step, retrieving compatible candidates at each stage via a hybrid embedding + colour-histogram scorer.

---

## What runs and what is for reference

After running `download_from_hf.py` the following work fully:

- **Streamlit app** (`src/app/streamlit_app.py`) — the main interactive demo
- **Evaluation** (`src/results/evaluate.py`, `evaluate_hybrid.py`)
- **EDA** (`src/eda/eda.py`) — figures and tables, pre-generated outputs already in `data/eda_output/`
- **Retrieval index rebuild** (`src/retrieval/make_embeddings.py`)
- **Retraining** (`src/ml/train.ipynb`) — training data is included in the HuggingFace dataset

The following are included for **methodology documentation**, but require access to the original raw data sources and will not run out of the box:

- `src/data_processing/sklad_mebliv/` — scraping and preprocessing for the Sklad Mebliv Ukrainian retailer: downloading product pages, parsing annotations, and normalising into the unified scene format
- `src/data_processing/deepfurn/process_annotations.ipynb` — annotation processing for DeepFurniture; note that the initial scene extraction and image-level preprocessing scripts are **not included** to avoid confusion with the full DeepFurniture pipeline, which operates on the original dataset from [byliu/DeepFurniture](https://huggingface.co/datasets/byliu/DeepFurniture)
- `src/data_processing/total/` — merging all sources into the unified catalog with `general_manifest.json`
- `src/ml/embeddings_for_training.ipynb` — ResNet50 embedding extraction over raw furniture images (prerequisite for triplet mining from scratch)
- `src/ml/build_triplets.py` — triplet construction from scratch (ready CSVs are downloadable from HuggingFace)
- `src/visualization/` — notebooks used to generate thesis figures

**Note on Wayfair:** `src/data_processing/wayfair/process_annotation.ipynb` is included for transparency. Wayfair was explored as a third data source early in the project but was ultimately dropped — the data proved difficult to normalise consistently and was bad itself. No Wayfair images or annotations appear in the final dataset, trained models, or any results.

---

## Repository structure

```
src/
  app/                    Streamlit retrieval app
  data_processing/        Per-source preprocessing notebooks (DeepFurn, Sklad Mebliv, Wayfair)
  eda/                    Exploratory data analysis script
  ml/                     Model definition, triplet builders, training notebook
  retrieval/              Embedding generation, retrieval logic, histogram filter
  results/                Evaluation scripts, result JSONs, and figures
  visualization/          Thesis figure notebooks
data/
  eda_output/             Pre-generated EDA figures and summary tables
  w_b_runs/               W&B run artifacts for the training runs (bedrooms + living rooms)
download_from_hf.py       Downloads all model + dataset artifacts from HuggingFace
requirements.txt
```

---

## Data

Two room types are covered.

Scenes are split into three non-overlapping scene-level sets: **golden** (18 %) reserved for final evaluation only, **train** (70 %), and **val** (12 %) for model selection.

Triplet construction: anchor = item from a scene → positive = different-category item from the **same** scene → negative = same category as positive from a **different** scene, selected from the [50th, 95th] cosine-distance percentile using ResNet50/ImageNet embeddings. Five negatives per anchor–positive pair.

**HuggingFace dataset:** [Darebal/furniture-catalog](https://huggingface.co/datasets/Darebal/furniture-catalog)  
Contains catalog images, triplet CSVs (train / val / golden), scene images, and CLIP embeddings.

---

## Model

**Architecture:** Siamese ResNet18 · embedding head: `Linear(512→256) → BN → ReLU → Linear(256→128)` · L2-normalised output · triplet margin loss (margin = 1.0) · AdamW + linear warmup + ReduceLROnPlateau.

Two separate models are trained — one per room type.

**HuggingFace model repo:** [Darebal/furniture-compatibility-siamese](https://huggingface.co/Darebal/furniture-compatibility-siamese)

### Results on the golden set

#### Bedrooms

| Model | Triplet acc. | MRR | Scene R@5 | Scene R@30 |
|---|---|---|---|---|
| ResNet18 (ImageNet, no fine-tuning) | 56.9 % | 0.479 | 5.0 % | 17.5 % |
| EfficientNet-B3 (ImageNet, no fine-tuning) | 54.0 % | 0.452 | 2.9 % | 10.8 % |
| **ResNet18 fine-tuned** | **81.4 %** | **0.754** | **36.9 %** | **63.1 %** |

#### Living rooms

| Model | Triplet acc. | MRR | Scene R@5 | Scene R@30 |
|---|---|---|---|---|
| ResNet18 (ImageNet, no fine-tuning) | 55.2 % | 0.474 | 15.4 % | 29.4 % |
| EfficientNet-B3 (ImageNet, no fine-tuning) | 53.7 % | 0.447 | 9.0 % | 22.1 % |
| **ResNet18 fine-tuned** | **73.0 %** | **0.635** | **14.5 %** | **50.0 %** |

The hybrid retriever (embedding + Bhattacharyya colour histogram, equal weighting) further improves scene recall: bedrooms reach **Scene R@10 = 40.0 %**, living rooms **Scene R@30 = 48.9 %**.

Training curves and distance-distribution plots are committed under `src/results/` and `data/w_b_runs/`.

---

## Setup

```bash
git clone https://github.com/Baredal/thesis_furnitures
cd thesis_furnitures
pip install -r requirements.txt

# Download model weights + retrieval artifacts + dataset from HuggingFace
python download_from_hf.py

# Run the Streamlit app
streamlit run src/app/streamlit_app.py
```

For wandb logging during retraining, create a `.env` file and paste your API key:
```
API_KEY_WANBD=your_wandb_api_key_here
```

---

## Retraining pipeline

All training data is available from HuggingFace after `download_from_hf.py`. The pipeline below assumes training data is already downloaded:

```bash
# 1. Build triplets (optional — ready CSVs are downloaded from HuggingFace)
python src/ml/build_triplets.py

# 2. Train (run notebook interactively)
#    src/ml/train.ipynb

# 3. Rebuild retrieval index from the new checkpoint
python src/retrieval/make_embeddings.py

# 4. Evaluate
python src/results/evaluate.py
python src/results/evaluate_hybrid.py
```

To reproduce from raw data (requires original source access), additionally run:
- `src/ml/embeddings_for_training.ipynb` — ResNet50 embedding extraction for negative mining
- Per-source notebooks in `src/data_processing/`

---

## EDA

```bash
python src/eda/eda.py
```

Pre-generated figures and tables are already committed in `data/eda_output/`.

---

## Citation

```bibtex
@thesis{Strus2026furniture,
  title  = {Visual Furnishings Compatibility Search Using Machine Learning},
  author = {Yaroslav-Dmytro Strus},
  school = {Ukrainian Catholic University},
  year   = {2026},
  type   = {Bachelor's Thesis}
}
```
