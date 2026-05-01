# Visual Furnishings Compatibility Search Using Machine Learning

Bachelor's thesis · Ukrainian Catholic University · 2026

**HuggingFace:** [Model weights](https://huggingface.co/Darebal/furniture-compatibility-siamese) · [Dataset](https://huggingface.co/datasets/Darebal/furniture-catalog)

---

## About

This repository is the full code base accompanying the bachelor's thesis of the same name. The thesis covers dataset construction, triplet-based metric learning, a hybrid retrieval system, and an interactive room-building demo. The primary contribution is the code and trained models; the written thesis document lives separately and is referenced in the citation below.

The project addresses visual furniture compatibility: given a selected furniture item, retrieve other items that are stylistically compatible with it. A Siamese ResNet18 is trained with triplet margin loss on a multi-source dataset covering two room types (bedrooms, living rooms). A Streamlit app lets a user build a room step-by-step, retrieving compatible candidates at each stage via a hybrid embedding + colour-histogram scorer.

### Problem and approach

Selecting furniture that looks good together is difficult without visual intuition or professional help. The goal of this project is to make that easier with a retrieval system that surfaces stylistically compatible items automatically.

The core idea is metric learning: a Siamese network is trained so that embeddings of compatible furniture items (items that appear together in the same real-world scene) are pulled closer together in embedding space, while incompatible items (same category but different scene) are pushed apart. At retrieval time, a query embedding is compared against all candidates in the catalog; the top results are items the model has learned to associate with that style.

A pure embedding similarity can favour items that look identical in colour rather than style. To counteract this, the final retrieval score blends the learned embedding similarity with a Bhattacharyya colour-histogram coefficient, with a user-adjustable weight. This lets the user bias results toward colour-matched items or toward style-matched items independently.

---

## App walkthrough

The Streamlit app guides a user through assembling a room step by step.

1. **Choose room type** — bedrooms or living rooms.
2. **Optional anchor photo** — upload your own furniture photo at step 0. The model embeds it on the fly and uses it as the starting point for all subsequent recommendations.
3. **Step through the chain** — for bedrooms the order is bed → small storage → large storage → table → chair/stool → curtain; for living rooms it is sofa → table → small storage → large storage → chair/stool → curtain.
4. **Pick or skip** — at each step five compatible candidates are shown, scored by the hybrid retriever. You can select one, skip the category, or shuffle for five more options.
5. **Adjust the scoring weight** — a sidebar slider lets you move between "style" (embedding-dominated) and "colour" (histogram-dominated) scoring in real time.
6. **Final room view** — all selected items are displayed together. Each item links to its original product page (where available) and can be reverse-searched via Google Lens. A collage of all selected items can be downloaded as a single JPEG.

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
- `src/data_processing/deepfurn/process_annotations.ipynb` — annotation processing for DeepFurniture; the furniture bounding-box cropping and image collection scripts are **not included** — they operate on the raw DeepFurniture source files and would not be meaningful without the full original dataset from [byliu/DeepFurniture](https://huggingface.co/datasets/byliu/DeepFurniture)
- `src/data_processing/total/` — merging all sources into the unified catalog with `general_manifest.json`
- `src/ml/embeddings_for_training.ipynb` — ResNet50 embedding extraction over raw furniture images (prerequisite for triplet mining from scratch)
- `src/ml/build_triplets.py` — triplet construction from scratch (ready CSVs are downloadable from HuggingFace)
- `src/visualization/` — notebooks used to generate thesis figures

**Note on data processing completeness:** For both DeepFurniture and Wayfair, the furniture extraction and image collection scripts are intentionally not included. Those steps depend on the original source files and reproducing them requires access to the raw datasets; including partial scripts would be more confusing than helpful. What is published covers annotation parsing and the normalisation logic only.

**Note on Wayfair:** `src/data_processing/wayfair/process_annotation.ipynb` is included for transparency. Wayfair was explored as a third data source early in the project but was ultimately dropped — the data proved difficult to normalise consistently. No Wayfair images or annotations appear in the final dataset, trained models, or any results.

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

## Pipeline

The full pipeline from raw data to the running app has seven stages:

1. **Per-source preprocessing** (`src/data_processing/`) — raw data from each source is downloaded, cleaned, and normalised into a unified scene format: each scene folder contains `annotation.json`, `scene_image.jpg`, and a `furnitures/` subfolder with one cropped image per item.
2. **Catalog aggregation** (`src/data_processing/total/`) — all sources are merged into `data/total/{room}/` with a single `general_manifest.json` listing every item with its category, source, scene, and image path.
3. **Embedding extraction for triplet mining** (`src/ml/embeddings_for_training.ipynb`) — a frozen ResNet50 (ImageNet V2) produces a 2048-dim feature vector for every furniture image. These are stored in `furniture_embeddings.json` and used only for negative selection during triplet construction.
4. **Triplet construction** (`src/ml/build_triplets.py`) — anchor/positive/negative triplets are mined scene-by-scene. Negatives are drawn from the [50th, 95th] cosine-distance percentile to ensure moderate difficulty. Scenes are split three ways: golden (18 %, held out for final evaluation), train (≈70 %), val (12 %, used for model selection).
5. **Siamese network training** (`src/ml/train.ipynb`) — a ResNet18 backbone with a two-layer embedding head is fine-tuned with triplet margin loss. Best checkpoint saved by val loss.
6. **Retrieval index generation** (`src/retrieval/make_embeddings.py`) — the trained model encodes every catalog item into a 128-dim embedding. Embeddings, item metadata, and colour histograms are saved to `data/retrieval_data/{room}/`.
7. **Streamlit app** (`src/app/streamlit_app.py`) — loads the retrieval index at startup and serves step-by-step room-building recommendations.

---

## Data

Two room types are covered: **bedrooms** and **living rooms**.

### Sources

| Source | Rooms | Scenes |
|---|---|---|
| [DeepFurniture](https://huggingface.co/datasets/byliu/DeepFurniture) | bedrooms + living rooms | 801 per room type |
| Sklad Mebliv (Ukrainian online retailer) | bedrooms only | 73 |

DeepFurniture provides real interior photographs with bounding-box annotations for individual furniture items. Sklad Mebliv scenes were scraped from product pages; each page groups several items from the same collection, which serves as a proxy for compatibility.

### Categories

| Room | Categories |
|---|---|
| Bedrooms | bed, small\_storage, large\_storage, table, chair\_stool, curtain |
| Living rooms | sofa, small\_storage, large\_storage, table, chair\_stool, curtain |

### Scene split

Scenes are split at the scene level (no furniture item appears in more than one split) into three non-overlapping sets: **golden** (18 %) reserved for final evaluation only, **train** (≈70 %), and **val** (12 %) for model selection. The split is stratified per source so each source contributes the same proportions across all three sets.

### Triplet construction

Anchor = item from a scene → positive = different-category item from the **same** scene → negative = same category as positive from a **different** scene.

Negative difficulty is controlled by cosine distance between the positive and negative embeddings (computed with a frozen ResNet50). Only negatives in the [50th, 95th] distance percentile are eligible — easy negatives (clearly dissimilar) and extreme outliers are excluded. Five negatives are sampled per anchor–positive pair with a per-scene quota to prevent any single scene from dominating.

**HuggingFace dataset:** [Darebal/furniture-catalog](https://huggingface.co/datasets/Darebal/furniture-catalog)  
Contains catalog images, triplet CSVs (train / val / golden), scene images, and CLIP embeddings.

---

## Model

**Architecture:** Siamese ResNet18 · embedding head: `Linear(512→256) → BN → ReLU → Linear(256→128)` · L2-normalised 128-dim output · triplet margin loss (margin = 1.0).

Both the backbone and the embedding head are trained end-to-end. All weights are shared between the two branches of the Siamese network (the standard weight-sharing setup).

Two separate models are trained — one per room type — because the category sets differ and the visual style distributions are distinct.

**HuggingFace model repo:** [Darebal/furniture-compatibility-siamese](https://huggingface.co/Darebal/furniture-compatibility-siamese)

### Training configuration

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Backbone LR | head LR × 0.1 (differential LR) |
| LR schedule | Linear warmup → ReduceLROnPlateau |
| Gradient clipping | max\_norm = 5.0 |
| Mixed precision | GradScaler (FP16) |
| Early stopping | patience on val loss |
| Experiment tracking | Weights & Biases (`furnishings_compatibility`) |

### Results on the golden set

#### Bedrooms

| Model | Triplet acc. | Margin sat. | Dist. gap | MRR | Scene R@5 | Scene R@10 | Scene R@30 | Compat R@10 |
|---|---|---|---|---|---|---|---|---|
| ResNet18 (ImageNet) | 56.9 % | 0.0 % | 0.021 | 0.479 | 5.0 % | 8.8 % | 17.5 % | 5.5 % |
| EfficientNet-B3 (ImageNet) | 54.0 % | 0.0 % | 0.012 | 0.452 | 2.9 % | 3.8 % | 10.8 % | 3.7 % |
| **ResNet18 fine-tuned** | **81.4 %** | **36.4 %** | **0.641** | **0.754** | **36.9 %** | **45.6 %** | **63.1 %** | **22.5 %** |

#### Living rooms

| Model | Triplet acc. | Margin sat. | Dist. gap | MRR | Scene R@5 | Scene R@10 | Scene R@30 | Compat R@10 |
|---|---|---|---|---|---|---|---|---|
| ResNet18 (ImageNet) | 55.2 % | 0.0 % | 0.017 | 0.474 | 15.4 % | 19.1 % | 29.4 % | 7.3 % |
| EfficientNet-B3 (ImageNet) | 53.7 % | 0.0 % | 0.010 | 0.447 | 9.0 % | 13.4 % | 22.1 % | 5.5 % |
| **ResNet18 fine-tuned** | **73.0 %** | **27.4 %** | **0.458** | **0.635** | **14.5 %** | **25.0 %** | **50.0 %** | **9.7 %** |

*Margin sat. = fraction of triplets where the embedding gap exceeds the margin (1.0). Dist. gap = mean(neg dist) − mean(pos dist). Compat R@10 = fraction of all compatible catalog items recovered in the top 10.*

#### Hybrid retriever (embedding + colour histogram, 50 / 50 weight)

| Room | Scene R@5 | Scene R@10 | Scene R@30 | Compat P@5 | Compat R@10 |
|---|---|---|---|---|---|
| Bedrooms (fine-tuned) | 26.9 % | 40.0 % | 62.1 % | 7.6 % | 19.3 % |
| Living rooms (fine-tuned) | 16.0 % | 25.7 % | 48.9 % | 4.6 % | 10.0 % |

The colour component biases results toward colour-coherent candidates; the scene-recall trade-off is adjustable via the sidebar slider in the app (default: 0.8 embedding / 0.2 colour).

Training curves and distance-distribution plots are committed under `src/results/` and `data/w_b_runs/`.

### Metric definitions

- **Triplet accuracy** — fraction of golden-set triplets where `dist(anchor, positive) < dist(anchor, negative)`. Measures whether the model correctly ranks the compatible item above the incompatible one.
- **Margin satisfied** — fraction of triplets where `dist(anchor, negative) − dist(anchor, positive) > margin (1.0)`. Stricter than triplet accuracy: checks that compatible and incompatible items are separated by at least the full margin, not just ordered correctly.
- **Distance gap** — `mean(dist(anchor, negative)) − mean(dist(anchor, positive))` over the golden set. Larger gap means embeddings of compatible items are pulled substantially closer than incompatible ones.
- **Triplet R@K** — per-anchor, K negatives are available (one per triplet); the query is a hit if the positive ranks in the top K among those negatives. Measures rank quality within the hard-negative context of the triplet set.
- **MRR** (Mean Reciprocal Rank) — for each anchor, all items of the positive's category are ranked by embedding similarity; MRR is the mean of 1/rank for the true positive. Measures how highly the model ranks the correct compatible item in a full gallery retrieval.
- **Scene R@K** (Scene Recall at K) — for each anchor, a gallery of all catalog items (not just the triplet negative) is ranked; the query is a hit if any item from the same scene as the positive appears in the top K. Measures real-world retrieval quality at different cut-off depths.
- **Compat Precision@K / Compat Recall@K** — ground truth is expanded beyond the single triplet positive to all catalog items co-occurring in the anchor's scene. Precision@K = fraction of top-K retrievals that are scene-compatible; Recall@K = fraction of all compatible items recovered in the top K. These measure retrieval quality against the full scene-level compatibility signal rather than just the triplet pair.

---

## Retrieval system

At runtime, `FurnitureRetriever` scores every candidate in the target category against the items already selected by the user:

```
score = embed_weight × cosine_sim(embedding_query, embedding_candidate)
      + hist_weight  × bhattacharyya(histogram_query, histogram_candidate)
```

- **Embedding similarity** uses the fine-tuned ResNet18 output (128-dim, L2-normalised). High similarity means the model learned that these item styles co-occur in real rooms.
- **Bhattacharyya coefficient** is computed over RGB histograms (32 bins per channel, sqrt-normalised so the coefficient equals the dot product). High coefficient means similar dominant colours.
- When multiple items have already been selected, their scores are aggregated with a **recency bias**: the most recently chosen item is weighted 1.0, older items linearly down to 0.3. This lets the chain build coherently around the latest selection.
- The user controls `embed_weight` / `hist_weight` (sum = 1.0) via a sidebar slider. Default is 0.8 / 0.2 — style-dominant with a small colour correction.

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

Generates four figures and four summary tables per room type: item counts by category and source, scene-size distribution, three-way split breakdown, category-pair heatmap in train triplets, and negative-distance distributions. Pre-generated outputs are already committed in `data/eda_output/`.

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
