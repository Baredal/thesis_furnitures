# Visual Furnishings Compatibility Search Using Machine Learning

Bachelor's thesis · Ukrainian Catholic University · 2026

**HuggingFace:** [Model weights](https://huggingface.co/Darebal/furniture-compatibility-siamese) · [Dataset](https://huggingface.co/datasets/Darebal/furniture-catalog)

---

## About

This repository is the full code base accompanying the bachelor's thesis of the same name. The thesis covers dataset construction, triplet-based metric learning, a hybrid retrieval system, and an interactive room-building demo. The primary contribution is the code and trained models; the written thesis document lives separately and is referenced in the citation below.

The project addresses visual furniture compatibility: given a selected furniture item, retrieve other items that are stylistically compatible with it. A Siamese ResNet18 is trained with triplet margin loss on a multi-source dataset covering two room types (bedrooms, living rooms). A Streamlit app lets a user build a room step-by-step, retrieving compatible candidates at each stage via a hybrid embedding + colour-histogram scorer.

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

**Note on data processing completeness:** In the case of DeepFurniture, not all scripts for extracting furniture and collecting images are included, and not all of them correspond 100% to the final dataset. This stage depends on the source files, and recreating them requires access to the raw datasets, which are very large. Additionally, the final dataset was compiled over many attempts, relabeled, and tested, which took a significant amount of time. Overall, the process involved searching the code for scenes with the appropriate number of furniture items (excluding trash), extracting scenes, relabeling, and unifying them for the training dataset.

**Note on Wayfair:** `src/data_processing/wayfair/process_annotation.ipynb` is included for transparency. Wayfair was explored as a third data source early in the project but was ultimately dropped — the data proved difficult to normalise consistently. No Wayfair images or annotations appear in the final dataset, trained models, or any results.

---

## Repository structure

```
src/
  app/
    streamlit_app.py              Interactive room-building demo — main entry point

  data_processing/
    deepfurn/
      process_annotations.ipynb   Normalise DeepFurniture bounding-box annotations
                                  into the unified scene format
      [other notebooks]           Scene extraction scripts — not included (see note above)
    sklad_mebliv/
      download_data.ipynb         Scrape Sklad Mebliv product pages
      parse_data.py / .ipynb      Parse HTML → per-item annotation JSON
      preprocess_data.ipynb       Crop furniture images; map Ukrainian labels to categories
      process_annotation.ipynb    Final normalisation to unified scene format
    total/
      move_analyse_transform.ipynb  Merge all sources → data/total/{room}/general_manifest.json
      normalize_images.py           Resize and standardise catalog images
    wayfair/
      process_annotation.ipynb    Annotation processing (source dropped — reference only)

  eda/
    eda.py                        Figures + tables: item counts, scene sizes, split
                                  breakdown, category-pair heatmap, distance distributions

  ml/
    model.py                      SiameseResNet18 and variants (ResNet34/50, EfficientNet
                                  B0–B7); shared embedding head architecture
    build_triplets.py             Mine anchor/positive/negative triplets from embeddings;
                                  produce golden/train/val scene split
    embeddings_for_training.ipynb Extract ResNet50 (ImageNet V2) features for negative
                                  mining — prerequisite for build_triplets.py
    train.ipynb                   Fine-tune SiameseResNet18 with triplet margin loss;
                                  W&B experiment tracking

  retrieval/
    retrieval_logic.py            FurnitureRetriever class — hybrid embedding + colour
                                  scorer with recency-weighted multi-item aggregation
    histogram_filter.py           RGB histogram (32 bins/channel); sqrt-encodes so
                                  Bhattacharyya coefficient = dot product
    make_embeddings.py            Encode full catalog → retrieval_embeddings.npz,
                                  retrieval_index.json, retrieval_histograms_bc.npz

  results/
    evaluate.py                   Triplet accuracy, MRR, Scene R@K, Compat P/R@K
                                  on the golden set (embedding-only scorer)
    evaluate_hybrid.py            Same metrics with the hybrid scorer
    {room}_results.json           Pre-computed evaluation results (embedding-only)
    {room}_results_hybrid.json    Pre-computed evaluation results (hybrid)
    *.png                         Precision/recall and distance-distribution plots

  visualization/
    fig1_grid.ipynb – fig5_weights.ipynb  Thesis figure notebooks
    comparison_visualization.ipynb        Side-by-side model comparison figures

data/
  eda_output/             Pre-generated EDA figures and summary tables
  w_b_runs/               W&B run artifacts for both room types

download_from_hf.py       Download models + dataset from HuggingFace
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
git lfs install
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

To reproduce from proccessed data creation triplets (requires processed data), additionally run:
- `src/ml/embeddings_for_training.ipynb` — ResNet50 embedding extraction for negative mining
- Per-source notebooks in `src/data_processing/`
- [Kaggle processed data](https://www.kaggle.com/datasets/darebal/furnishings-dataset)
- You will have to download, unarchive and move **processed_data** into **data/** folder

Processed is standardised into a unified directory structure regardless of its original source (e.g., DeepFurniture or Sklad Mebliv) or room type. The directory hierarchy for a processed scene looks like this:

```text
processed_data/
  {room_type}/             # e.g., bedrooms
    {source}/              # e.g., deepfurn
      {scene_id}/          # e.g., deepfurn_0000
        annotations.json   # Scene metadata and labels
        scene_image.jpg    # The full, uncropped original image
        furnitures/        # Directory containing individually cropped items
          {furniture_id}.jpg 
          ...

```
The annotations.json file within every scene directory follows a strict schema that links each cropped image inside the furnitures/ folder to its corresponding category:

```
{
  "scene_name": "deepfurn_0000",
  "furnitures": [
    {
      "furniture_id": "2839115",
      "category": "chair_stool"
    },
    {
      "furniture_id": "6801",
      "category": "table"
    },
    {
      "furniture_id": "2830190",
      "category": "curtain"
    },
    {
      "furniture_id": "2875981",
      "category": "bed"
    },
    {
      "furniture_id": "3516687",
      "category": "small_storage"
    }
  ]
}

```
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
