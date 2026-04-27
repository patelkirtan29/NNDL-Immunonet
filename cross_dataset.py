"""
ImmunoNet — Step 8: Cross-Dataset Validation (GSE126030)
==========================================================
This script:
  1. Extracts and loads GSE126030 10x Genomics count matrices from the RAW tar
  2. Combines samples, preprocesses to match GSE108989 pipeline
  3. Assigns labels from experimental design (resting vs activated)
  4. Tests all trained models (classical + DL) on this new data
  5. Reports cross-dataset generalization performance

GSE126030 contains ~50,000 T-cells from 4 tissues (blood, lung,
lymph node, bone marrow), each with resting and activated conditions.
Labels come from the experimental design, NOT from clustering.

Since our GSE108989 model was trained on 6 classes (Naive, Effector,
Exhausted, Treg, Th1-like, Other_CD4), we map:
  - Resting T-cells → should be predicted as Naive
  - Activated T-cells → should be predicted as Effector/Th1-like

This tests whether models learn universal T-cell biology.

Requirements: pip install numpy pandas scanpy scipy scikit-learn tensorflow matplotlib seaborn

Input:
  - GSE126030_RAW.tar (10x Genomics count matrices)
  - GSE126030_family.soft (or .soft.gz)
  - Trained models from results/
  - step3_*.npy preprocessing artifacts

Output: results/cross_dataset/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import mmread
from scipy.sparse import issparse
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler
import json
import os
import tarfile
import gzip
import glob
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION — Update these paths
# ============================================================
TAR_FILE = "/Users/kirtan/Downloads/GSE126030_RAW.tar"
SOFT_FILE = "/Users/kirtan/Downloads/GSE126030_family.soft"  # or .soft.gz
EXTRACT_DIR = "gse126030_extracted"
OUTPUT_DIR = "results/cross_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

# ============================================================
# STEP 8A: Extract the RAW tar file
# ============================================================
print("=" * 60)
print("STEP 8A: Extracting GSE126030_RAW.tar...")
print("=" * 60)

if not os.path.exists(os.path.join(EXTRACT_DIR, "_extracted_done")):
    start = time.time()
    with tarfile.open(TAR_FILE, "r") as tar:
        tar.extractall(path=EXTRACT_DIR)
    # Mark as done so we don't re-extract
    open(os.path.join(EXTRACT_DIR, "_extracted_done"), "w").close()
    print(f"  Extracted in {time.time()-start:.1f}s")
else:
    print(f"  Already extracted, skipping...")

# List what we got
extracted_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, "*")))
print(f"  Found {len(extracted_files)} files:")
for f in extracted_files[:20]:
    print(f"    {os.path.basename(f)}")
if len(extracted_files) > 20:
    print(f"    ... and {len(extracted_files) - 20} more")

# ============================================================
# STEP 8B: Parse sample metadata from SOFT file
# ============================================================
print("\n" + "=" * 60)
print("STEP 8B: Parsing sample metadata from SOFT file...")
print("=" * 60)

# Parse SOFT file for sample info (tissue, condition, donor)
samples = {}
current_sample = None

soft_path = SOFT_FILE
if soft_path.endswith(".gz"):
    import gzip as gz
    open_func = lambda p: gz.open(p, "rt", errors="replace")
else:
    open_func = lambda p: open(p, "r", errors="replace")

with open_func(soft_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("^SAMPLE"):
            current_sample = line.split("=")[-1].strip()
            samples[current_sample] = {}
        elif current_sample and line.startswith("!Sample_title"):
            samples[current_sample]["title"] = line.split("=", 1)[-1].strip()
        elif current_sample and line.startswith("!Sample_source_name"):
            samples[current_sample]["source"] = line.split("=", 1)[-1].strip()
        elif current_sample and line.startswith("!Sample_supplementary_file"):
            val = line.split("=", 1)[-1].strip()
            if "supplementary_files" not in samples[current_sample]:
                samples[current_sample]["supplementary_files"] = []
            samples[current_sample]["supplementary_files"].append(val)

print(f"  Found {len(samples)} samples:")
for gsm, info in samples.items():
    title = info.get("title", "N/A")
    source = info.get("source", "N/A")
    print(f"    {gsm}: {title} | {source}")

# Extract condition (resting vs activated) and tissue from title
for gsm, info in samples.items():
    title = info.get("title", "").lower()
    source = info.get("source", "").lower()

    # Determine condition
    if "activated" in title or "activated" in source or "anti-cd3" in title:
        info["condition"] = "Activated"
    elif "resting" in title or "unstimulated" in title:
        info["condition"] = "Resting"
    else:
        info["condition"] = "Unknown"

    # Determine tissue
    combined = title + " " + source
    if "lung" in combined:
        info["tissue"] = "Lung"
    elif "lymph" in combined or "ln " in combined:
        info["tissue"] = "Lymph_Node"
    elif "bone" in combined or "bm " in combined:
        info["tissue"] = "Bone_Marrow"
    elif "blood" in combined or "pbmc" in combined:
        info["tissue"] = "Blood"
    else:
        info["tissue"] = "Unknown"

print(f"\n  Parsed conditions and tissues:")
for gsm, info in samples.items():
    print(f"    {gsm}: {info.get('condition', '?')} | {info.get('tissue', '?')} | {info.get('title', '?')}")

# ============================================================
# STEP 8C: Load 10x count matrices
# ============================================================
print("\n" + "=" * 60)
print("STEP 8C: Loading 10x count matrices...")
print("=" * 60)

def load_filtered_matrix(filepath, sample_id):
    """
    Load a GSE126030 .filtered.matrix.txt.gz file.
    Format:
      Row 0 (header): Accession  Gene  barcode1  barcode2  ...
      Row 1+:         ENSG...    GZMB  0         5         ...
    Returns: expression matrix (cells x genes), gene names, barcodes
    """
    try:
        df = pd.read_csv(filepath, sep="\t", compression="gzip")

        # First two columns are Accession (Ensembl ID) and Gene (symbol)
        gene_names = df["Gene"].values.tolist()
        barcodes = df.columns[2:].tolist()  # cell barcodes

        # Expression matrix: genes x cells → transpose to cells x genes
        matrix = df.iloc[:, 2:].values.T.astype(np.float32)

        return matrix, gene_names, barcodes
    except Exception as e:
        print(f"    Error loading {sample_id}: {e}")
        return None, None, None


# Load all samples
all_matrices = []
all_barcodes = []
all_conditions = []
all_tissues = []
sample_gene_names = None

# Find extracted .gz files and match to GSM IDs
extracted_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, "*.gz")))
gsm_to_file = {}
for f in extracted_files:
    basename = os.path.basename(f)
    # Extract GSM ID from filename like "GSM3589406_PP001swap.filtered.matrix.txt.gz"
    gsm_id = basename.split("_")[0]
    gsm_to_file[gsm_id] = f

print(f"  Found {len(gsm_to_file)} sample files to load")

for gsm, info in samples.items():
    filepath = gsm_to_file.get(gsm)
    if filepath is None:
        print(f"\n  {gsm}: No matching file found, skipping")
        continue

    print(f"\n  Loading {gsm} ({info.get('condition', '?')}, {info.get('tissue', '?')})...")

    matrix, genes, barcodes = load_filtered_matrix(filepath, gsm)

    if matrix is not None:
        n_cells = matrix.shape[0]
        n_genes = matrix.shape[1]
        print(f"    Loaded: {n_cells} cells x {n_genes} genes")

        all_matrices.append(matrix)
        all_barcodes.extend([f"{gsm}_{b}" for b in barcodes])
        all_conditions.extend([info.get("condition", "Unknown")] * n_cells)
        all_tissues.extend([info.get("tissue", "Unknown")] * n_cells)

        if sample_gene_names is None:
            sample_gene_names = genes
        else:
            # Verify gene order is consistent across samples
            if genes != sample_gene_names:
                print(f"    WARNING: Gene order differs from first sample — realigning")
                # Realign to first sample's gene order
                gene_to_idx = {g: i for i, g in enumerate(genes)}
                aligned = np.zeros((n_cells, len(sample_gene_names)), dtype=np.float32)
                for i, g in enumerate(sample_gene_names):
                    if g in gene_to_idx:
                        aligned[:, i] = matrix[:, gene_to_idx[g]]
                all_matrices[-1] = aligned  # replace with aligned version
    else:
        print(f"    Could not load — skipping")

if len(all_matrices) == 0:
    print("\n  ERROR: No samples could be loaded!")
    print("  Check that the tar extracted .gz files into:", EXTRACT_DIR)
    print("  Expected files like: GSM3589406_PP001swap.filtered.matrix.txt.gz")
    exit()

# Combine all samples
print(f"\n  Combining all samples...")
X_cross = np.vstack(all_matrices).astype(np.float32)
conditions = np.array(all_conditions)
tissues = np.array(all_tissues)

print(f"  Combined: {X_cross.shape[0]} cells x {X_cross.shape[1]} genes")
print(f"  Conditions: {pd.Series(conditions).value_counts().to_dict()}")
print(f"  Tissues: {pd.Series(tissues).value_counts().to_dict()}")

# ============================================================
# STEP 8D: Preprocess to match GSE108989 pipeline
# ============================================================
print("\n" + "=" * 60)
print("STEP 8D: Preprocessing to match training data pipeline...")
print("=" * 60)

# Load the gene names used in training
train_gene_names = np.load("step3_gene_names.npy", allow_pickle=True)

# Find overlapping genes between GSE126030 and our selected training genes
cross_gene_set = set(sample_gene_names)
train_gene_set = set(train_gene_names)
overlap = cross_gene_set.intersection(train_gene_set)

print(f"  Training genes: {len(train_gene_names)}")
print(f"  Cross-dataset genes: {len(sample_gene_names)}")
print(f"  Overlapping genes: {len(overlap)}")
print(f"  Overlap percentage: {len(overlap)/len(train_gene_names)*100:.1f}%")

# Build aligned matrix: same columns as training data, in same order
# Missing genes get filled with zeros
X_aligned = np.zeros((X_cross.shape[0], len(train_gene_names)), dtype=np.float32)

gene_to_idx_cross = {g: i for i, g in enumerate(sample_gene_names)}

n_matched = 0
for i, gene in enumerate(train_gene_names):
    if gene in gene_to_idx_cross:
        X_aligned[:, i] = X_cross[:, gene_to_idx_cross[gene]]
        n_matched += 1

print(f"  Matched {n_matched}/{len(train_gene_names)} genes ({n_matched/len(train_gene_names)*100:.1f}%)")

# Apply same preprocessing: log2(count + 1) then z-score
print(f"  Applying log2(x+1) transformation...")
X_aligned = np.log2(X_aligned + 1)

print(f"  Applying z-score normalization (using training scaler)...")
scaler_path = "step3_scaler.pkl"
if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_preprocessed = scaler.transform(X_aligned)
    print(f"  Used saved scaler from training")
    nan_count = np.isnan(X_preprocessed).sum()
    if nan_count > 0:
        print(f"  Fixing {nan_count} NaN values (from missing genes)...")
        X_preprocessed = np.nan_to_num(X_preprocessed, nan=0.0)
        print(f"  NaN after fix: {np.isnan(X_preprocessed).sum()}")
else:
    # Fallback: normalize independently
    scaler = StandardScaler()
    X_preprocessed = scaler.fit_transform(X_aligned)
    print(f"  Warning: Training scaler not found, normalized independently")
    nan_count = np.isnan(X_preprocessed).sum()
    if nan_count > 0:
        print(f"  Fixing {nan_count} NaN values (from missing genes)...")
        X_preprocessed = np.nan_to_num(X_preprocessed, nan=0.0)
        print(f"  NaN after fix: {np.isnan(X_preprocessed).sum()}")

print(f"  Final preprocessed shape: {X_preprocessed.shape}")

# ============================================================
# STEP 8E: Run trained models on cross-dataset
# ============================================================
print("\n" + "=" * 60)
print("STEP 8E: Running trained models on GSE126030...")
print("=" * 60)

with open("step3_label_mapping.json") as f:
    label_map = json.load(f)
class_names = [label_map[str(i)] for i in range(len(label_map))]

# Expected mapping for evaluation:
# Resting → Naive (the model should predict resting cells as Naive)
# Activated → Effector or Th1-like (activated cells should look like effectors)
expected_mapping = {
    "Resting": ["Naive"],
    "Activated": ["Effector", "Th1-like"]
}

cross_results = {}

# Helper function
def run_model_cross(model, model_name, X_data, is_cnn=False, is_keras=False):
    """Run a model on cross-dataset and analyze predictions."""

    print(f"\n  Running {model_name}...")

    if is_cnn:
        X_input = X_data.reshape(-1, X_data.shape[1], 1)
    else:
        X_input = X_data

    if is_keras:
        y_proba = model.predict(X_input, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
    else:
        y_pred = model.predict(X_input)

    pred_names = np.array([label_map[str(p)] for p in y_pred])

    # Overall prediction distribution
    pred_dist = pd.Series(pred_names).value_counts()
    print(f"    Prediction distribution:")
    for cls, count in pred_dist.items():
        pct = count / len(pred_names) * 100
        print(f"      {cls:<12}: {count:>6} ({pct:.1f}%)")

    # Per-condition analysis
    condition_analysis = {}
    for cond in ["Resting", "Activated"]:
        mask = conditions == cond
        if mask.sum() == 0:
            continue

        cond_preds = pred_names[mask]
        cond_dist = pd.Series(cond_preds).value_counts()
        expected = expected_mapping.get(cond, [])

        # What percentage of predictions match expected classes?
        correct = sum(cond_dist.get(cls, 0) for cls in expected)
        total = mask.sum()
        alignment_pct = correct / total * 100

        condition_analysis[cond] = {
            "total_cells": int(total),
            "prediction_distribution": cond_dist.to_dict(),
            "expected_classes": expected,
            "alignment_pct": float(alignment_pct)
        }

        print(f"\n    {cond} cells ({total}):")
        print(f"      Expected: {expected}")
        for cls, count in cond_dist.items():
            marker = "✓" if cls in expected else " "
            pct = count / total * 100
            print(f"      {marker} {cls:<12}: {count:>5} ({pct:.1f}%)")
        print(f"      Alignment with expected: {alignment_pct:.1f}%")

    # Per-tissue analysis
    tissue_analysis = {}
    for tissue in np.unique(tissues):
        mask = tissues == tissue
        if mask.sum() == 0:
            continue
        tissue_preds = pred_names[mask]
        tissue_dist = pd.Series(tissue_preds).value_counts()
        tissue_analysis[tissue] = tissue_dist.to_dict()

    result = {
        "model": model_name,
        "total_cells": len(pred_names),
        "prediction_distribution": pred_dist.to_dict(),
        "condition_analysis": condition_analysis,
        "tissue_analysis": tissue_analysis
    }

    cross_results[model_name] = result
    return pred_names


# --- Load and run all models ---

# Classical models
model_paths = {
    "Logistic Regression": "results/logistic_regression/model.pkl",
    "SVM (Linear)": "results/svm_linear/model.pkl",
    "SVM (RBF)": "results/svm_rbf/model.pkl",
    "Random Forest": "results/random_forest/model.pkl",
    "XGBoost": "results/xgboost/model.pkl",
}

for name, path in model_paths.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        run_model_cross(model, name, X_preprocessed)
    else:
        print(f"  {name}: model not found at {path}, skipping")

# DL models
from tensorflow import keras

dl_models = {
    "MLP": ("results/mlp/model.keras", False),
    "1D-CNN": ("results/cnn_1d/model.keras", True),
}

for name, (path, is_cnn) in dl_models.items():
    if os.path.exists(path):
        model = keras.models.load_model(path)
        run_model_cross(model, name, X_preprocessed, is_cnn=is_cnn, is_keras=True)
    else:
        print(f"  {name}: model not found at {path}, skipping")

# ============================================================
# STEP 8F: Generate cross-dataset figures
# ============================================================
print("\n" + "=" * 60)
print("STEP 8F: Generating cross-dataset figures...")
print("=" * 60)

# --- Figure 1: Alignment scores across models ---
fig, ax = plt.subplots(figsize=(12, 6))

model_names_list = []
resting_scores = []
activated_scores = []

for name, result in cross_results.items():
    model_names_list.append(name)
    ca = result["condition_analysis"]
    resting_scores.append(ca.get("Resting", {}).get("alignment_pct", 0))
    activated_scores.append(ca.get("Activated", {}).get("alignment_pct", 0))

x = np.arange(len(model_names_list))
width = 0.35

bars1 = ax.bar(x - width/2, resting_scores, width, label="Resting → Naive", color="#3498db")
bars2 = ax.bar(x + width/2, activated_scores, width, label="Activated → Effector/Th1-like", color="#e74c3c")

ax.set_ylabel("Alignment %", fontsize=12)
ax.set_title("Cross-Dataset Validation: GSE126030 Prediction Alignment\n(Higher = model learned universal T-cell biology)",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_names_list, rotation=25, ha="right", fontsize=9)
ax.legend(fontsize=10)
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3)

for i, (r, a) in enumerate(zip(resting_scores, activated_scores)):
    ax.text(i - width/2, r + 1, f"{r:.0f}%", ha="center", fontsize=8)
    ax.text(i + width/2, a + 1, f"{a:.0f}%", ha="center", fontsize=8)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "cross_dataset_alignment.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")

# --- Figure 2: Prediction distribution per condition (best model) ---
if cross_results:
    # Pick the model with highest average alignment
    best_model_name = max(cross_results, key=lambda n:
        np.mean([cross_results[n]["condition_analysis"].get(c, {}).get("alignment_pct", 0)
                 for c in ["Resting", "Activated"]]))

    best_result = cross_results[best_model_name]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, cond in enumerate(["Resting", "Activated"]):
        if cond not in best_result["condition_analysis"]:
            continue

        ca = best_result["condition_analysis"][cond]
        dist = ca["prediction_distribution"]
        expected = ca["expected_classes"]

        classes = list(dist.keys())
        counts = list(dist.values())
        colors = ["#2ecc71" if c in expected else "#e74c3c" for c in classes]

        axes[idx].bar(classes, counts, color=colors, edgecolor="white")
        axes[idx].set_title(f"{cond} Cells ({ca['total_cells']} total)\nAlignment: {ca['alignment_pct']:.1f}%",
                           fontsize=11, fontweight="bold")
        axes[idx].set_ylabel("Count")
        axes[idx].tick_params(axis="x", rotation=30)

        for i, (cls, count) in enumerate(zip(classes, counts)):
            pct = count / ca["total_cells"] * 100
            axes[idx].text(i, count + 20, f"{pct:.0f}%", ha="center", fontsize=9)

    plt.suptitle(f"Cross-Dataset Predictions — {best_model_name}\n(Green = expected, Red = unexpected)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cross_dataset_predictions.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# --- Figure 3: Tissue breakdown ---
if cross_results and best_model_name in cross_results:
    tissue_data = cross_results[best_model_name]["tissue_analysis"]

    if tissue_data:
        all_classes = sorted(set(c for t in tissue_data.values() for c in t.keys()))
        tissue_names = sorted(tissue_data.keys())

        fig, axes = plt.subplots(1, len(tissue_names), figsize=(5*len(tissue_names), 5))
        if len(tissue_names) == 1:
            axes = [axes]

        for idx, tissue in enumerate(tissue_names):
            dist = tissue_data[tissue]
            classes = list(dist.keys())
            counts = list(dist.values())

            axes[idx].bar(classes, counts, color="#3498db", edgecolor="white")
            axes[idx].set_title(f"{tissue}", fontsize=11, fontweight="bold")
            axes[idx].set_ylabel("Count")
            axes[idx].tick_params(axis="x", rotation=45, labelsize=8)

        plt.suptitle(f"Predictions per Tissue — {best_model_name}",
                     fontsize=13, fontweight="bold", y=1.03)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "cross_dataset_tissue.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

# ============================================================
# STEP 8G: Save all results
# ============================================================
print("\n" + "=" * 60)
print("STEP 8G: Saving results...")
print("=" * 60)

with open(os.path.join(OUTPUT_DIR, "cross_dataset_results.json"), "w") as f:
    json.dump(cross_results, f, indent=2, default=str)
print(f"  Saved: cross_dataset_results.json")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 8 COMPLETE — Cross-Dataset Validation")
print("=" * 60)
print(f"""
  What we tested:
    Trained models on GSE108989 (colorectal cancer, Smart-seq2)
    Tested on GSE126030 (healthy donors, 10x Genomics)
    Everything is different: patients, tissues, health, protocol

  Evaluation logic:
    Resting T-cells should be predicted as → Naive
    Activated T-cells should be predicted as → Effector or Th1-like

  Results:""")

for name, result in cross_results.items():
    ca = result["condition_analysis"]
    r_pct = ca.get("Resting", {}).get("alignment_pct", 0)
    a_pct = ca.get("Activated", {}).get("alignment_pct", 0)
    avg = (r_pct + a_pct) / 2
    print(f"    {name:<22}: Resting→Naive={r_pct:.0f}%, Activated→Effector={a_pct:.0f}%, Avg={avg:.0f}%")

print(f"""
  Files in {OUTPUT_DIR}/:
    cross_dataset_alignment.png   — Alignment scores across all models
    cross_dataset_predictions.png — Prediction distribution for best model
    cross_dataset_tissue.png      — Breakdown by tissue
    cross_dataset_results.json    — All metrics

  Next step: Step 9 (Final comparison table + statistical testing)
""")