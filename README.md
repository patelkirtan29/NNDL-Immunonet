# ImmunoNet — Classifying T-Cell States from Gene Expression

CSCI 4366/6366 · Neural Networks and Deep Learning  
Authors: Kirtan Patel · Swapnaneel Chatterjee

Classifies T-cell functional states (Naive, Effector, Exhausted, Th1-like, Treg, Other CD4) from scRNA-seq gene expression data and evaluates cross-dataset generalization across sequencing technologies.

---

## Requirements

Python 3.11 is required.

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy umap-learn scanpy tensorflow keras joblib
```

All experiments were run with:
- `tensorflow==2.21.0` / `keras==3.14.0`
- `numpy==2.4.4`
- `scikit-learn` (latest)

---

## Data Download

Download the following files from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/) and place them in your `~/Downloads/` folder (or update the hardcoded paths in the scripts):

| Dataset | Files needed | GEO Accession |
|---|---|---|
| Source (GSE108989) | `GSE108989_CRC.TCell.S11138.TPM.txt` | [GSE108989](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108989) |
| Source (GSE108989) | `GSE108989_family.soft` | [GSE108989](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108989) |
| Target 2 (GSE99254) | `GSE99254_NSCLC.TCell.S12346.TPM.txt` | [GSE99254](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE99254) |

> The primary target dataset (GSE126030) raw `.gz` files are already included in `gse126030_extracted/`.

---

## Pipeline — Run in This Order

### Step 1 — Parse raw data (`preprocessing.py`)

Parses the GSE108989 SOFT file for cell labels and merges them with the TPM expression matrix.

**Before running:** open `preprocessing.py` and update the two path constants at the top:
```python
TPM_FILE  = "/path/to/GSE108989_CRC.TCell.S11138.TPM.txt"
SOFT_FILE = "/path/to/GSE108989_family.soft"
```

```bash
python preprocessing.py
```

**Outputs:**
```
step1_labels.csv          # cell annotations (UniqueCell_ID, majorCluster, ...)
step1_merged.csv          # full merged matrix (cells × genes + metadata)
step1_preview.csv         # first 5 rows sanity check
```

---

### Step 2 — Clean and filter (`cleaning.py`)

Removes noise cell types (MAIT, iNKT, double-negative, etc.) and maps `majorCluster` labels to 6 canonical T-cell states.

```bash
python cleaning.py
```

**Input:** `step1_merged.csv`  
**Outputs:**
```
step2_cleaned.csv              # filtered matrix (6,824 cells × genes + metadata)
step2_class_distribution.png   # class balance bar chart
```

---

### Step 3 — Feature selection + train/test split (`step3_v2.py`)

Selects 3,000 highly variable genes using Fano factor, force-includes key T-cell marker genes, z-score normalizes, runs PCA, and creates the final train/test arrays.

```bash
python step3_v2.py
```

**Input:** `step2_cleaned.csv`  
**Outputs:**
```
step3_X_train.npy             # (5,459 × 3000) source train features
step3_X_test.npy              # (1,365 × 3000) source test features
step3_y_train.npy             # train labels (integer encoded)
step3_y_test.npy              # test labels
step3_gene_names.npy          # selected HVG names (3000,)
step3_gene_fano.npy           # Fano scores for selected genes
step3_class_weights.npy       # inverse-frequency class weights
step3_label_mapping.json      # int → class name mapping
step3_scaler.pkl              # fitted StandardScaler
step3_pca.pkl                 # fitted PCA (50 components)
step3_config.json             # preprocessing config snapshot
step3_X_train_pca.npy         # PCA-reduced train features
step3_X_test_pca.npy          # PCA-reduced test features
```

> `step3.py` is the original v1 (variance-based HVG). Use `step3_v2.py` for all experiments — it is the final version.

---

### Step 3b — Preprocess primary target dataset (`preprocess_target_v2.py`)

Preprocesses GSE126030 (10x Genomics) independently, removes dead genes from both source and target, and overwrites the source `.npy` files with the cleaned feature set.

```bash
python preprocess_target_v2.py
```

**Inputs:** `gse126030_extracted/*.gz`, `step3_gene_names.npy`, `step3_X_train.npy`, `step3_X_test.npy`  
**Outputs:**
```
gse126030_preprocessed_v2.npy  # target matrix (aligned, independently standardised)
step3_X_train.npy              # overwritten — dead genes removed (→ 2701 genes)
step3_X_test.npy               # overwritten — dead genes removed
step3_gene_names.npy           # overwritten — dead genes removed
step3_gene_fano.npy            # overwritten — dead genes removed
dead_gene_mask.npy             # boolean mask (True = gene kept)
target_scaler.pkl              # target StandardScaler
```

> **Important:** run this before any experiment that uses the GSE126030 target. All downstream `.npy` files will now have 2,701 genes (not 3,000).

---

### Step 3c — Preprocess validation target dataset (`preprocess_gse99254.py`)

Preprocesses GSE99254 (NSCLC Smart-seq2 dataset, same technology as source — no CORAL needed).

**Before running:** open `preprocess_gse99254.py` and update:
```python
TPM_FILE = "/path/to/GSE99254_NSCLC.TCell.S12346.TPM.txt"
ROOT     = "/path/to/this/repo"
```

```bash
python preprocess_gse99254.py
```

**Inputs:** `GSE99254_NSCLC.TCell.S12346.TPM.txt`, `step3_gene_names.npy`  
**Outputs:**
```
gse99254_preprocessed.npy      # (12,346 × 2701) aligned target features
gse99254_cell_meta.csv         # cell ID, tissue, coarse type, patient
gse99254_coarse_labels.npy     # integer labels from cell ID parsing
gse99254_label_mapping.json    # int → class name
```

---

### Step 4 — Exploratory data analysis (`eda.py`)

Generates UMAP, marker gene heatmaps, boxplots, and correlation figures.

```bash
python eda.py
```

**Inputs:** `step3_*.npy`, `step2_cleaned.csv`  
**Outputs:** figures saved to `figures/` directory

---

### Step 5 — Logistic regression baseline (`LR.py`)

Trains a logistic regression with elastic net regularization and stratified 5-fold CV tuning.

```bash
python LR.py
```

**Inputs:** `step3_*.npy`  
**Outputs:** `results/logistic_regression/` — model, predictions, confusion matrix, ROC curves, per-class F1

---

### Step 6 — Deep learning models (`dl.py`)

Trains MLP, 1D-CNN, and a lightweight self-attention classifier sequentially.

```bash
python dl.py
```

**Inputs:** `step3_*.npy`  
**Outputs:**
```
results/mlp/                   # MLP model, metrics, training curves
results/cnn_1d/                # 1D-CNN model, metrics, training curves
results/attention/             # Self-attention model, attention weights
results/dl_comparison/         # Side-by-side comparison of all DL models
```

---

### Step 7 — Interpretability (`integrated_gradient.py`)

Computes Integrated Gradients for MLP and 1D-CNN to identify top genes per T-cell class and validates against known biological markers.

```bash
python integrated_gradient.py
```

**Inputs:** `results/mlp/model.keras`, `results/cnn_1d/model.keras`, `step3_*.npy`  
**Outputs:** `results/interpretability/` — IG heatmaps, top-gene rankings, biological validation JSON

---

### Step 8 — Cross-dataset generalization (`cross_dataset.py`)

Evaluates all trained classical and DL models on the GSE126030 target dataset.

```bash
python cross_dataset.py
```

**Inputs:** all trained models from `results/`, `gse126030_preprocessed_v2.npy`  
**Outputs:** `results/cross_dataset/` — alignment plot, prediction distributions, tissue breakdown, results JSON

---

## Experiments (Development Notebooks)

These Jupyter notebooks document the iterative model development. Run them in order using Jupyter Lab or Jupyter Notebook:

```bash
pip install jupyterlab
jupyter lab
```

| Folder | Description |
|---|---|
| `experiments/step0_baselines/` | Logistic regression + random forest baselines; run `run_baselines.py` or open `step0_baselines.ipynb` |
| `experiments/step1_mmd_net/` | MMD-Net domain adaptation architecture exploration |
| `experiments/step2_dann/` | DANN → improved DANN → CDAN-E progression |
| `experiments/step3_self_attention/` | GeneAttention v1 → v6 development; `06_gene_attention_v6.ipynb` is the final best model |
| `experiments/step4_transformer/` | Full transformer classifier (over-parameterized baseline) |
| `experiments/step5_vae/` | VAE pre-training + joint VAE-classifier |

### Key notebooks in `experiments/step3_self_attention/`:

| Notebook | What it does |
|---|---|
| `06_gene_attention_v6.ipynb` | **Final model** — GeneAttention v6 training |
| `07_explainability.ipynb` | Attention-weight interpretability + IG comparison |
| `08_gse99254_eval.ipynb` | Transfer evaluation on GSE99254 |
| `09_gse99254_indomain_cv.ipynb` | In-domain cross-validation on GSE99254 |
| `10_shared_hvg_transfer.ipynb` | HVG sweep for shared-gene transfer |

---

## Saved Best Model

The best model weights (GeneAttention v6) are saved at:

```
saved_models/gene_attention_v6.weights.h5
```

Load in the GeneAttention v6 notebook (`06_gene_attention_v6.ipynb`) to reproduce inference without retraining.

Ensemble weights (v5, three seeds) are at:

```
experiments/step3_self_attention/results/ensemble_v5/seed_42.weights.h5
experiments/step3_self_attention/results/ensemble_v5/seed_123.weights.h5
experiments/step3_self_attention/results/ensemble_v5/seed_7.weights.h5
```

---

## Model Definitions

`models.py` contains shared Keras model class definitions used across experiment notebooks (GeneAttention, encoder blocks, etc.). It is imported by the notebooks — do not run it directly.

---

## File Reference

```
preprocessing.py          Step 1 — parse GSE108989 raw data
cleaning.py               Step 2 — filter noise labels
step3_v2.py               Step 3 — HVG selection, normalization, train/test split  (USE THIS)
step3.py                  Step 3 v1 — original variance-based HVG (legacy, kept for reference)
preprocess_target_v2.py   Step 3b — GSE126030 target preprocessing + dead-gene removal
preprocess_gse99254.py    Step 3c — GSE99254 validation target preprocessing
eda.py                    Step 4 — EDA figures
LR.py                     Step 5 — logistic regression baseline
dl.py                     Step 6 — MLP, 1D-CNN, self-attention
integrated_gradient.py    Step 7 — IG interpretability
cross_dataset.py          Step 8 — cross-dataset generalization test
models.py                 Shared Keras model definitions (imported by notebooks)
experiments/              Iterative development notebooks (step0 → step5)
results/                  Outputs from steps 5–8 (metrics, plots, predictions)
saved_models/             Best model weights for direct inference
gse126030_extracted/      Raw GSE126030 10x .gz files
```
