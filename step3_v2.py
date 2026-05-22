"""
ImmunoNet — Step 3 v2: Improved Preprocessing Pipeline
========================================================
Key changes vs step3.py:
  1. HVG selection uses Fano factor (variance/mean) instead of raw variance.
     Raw variance favours highly-expressed housekeeping genes; Fano factor
     selects genes that vary MORE than expected given their mean — the
     standard metric for biologically informative scRNA-seq HVGs.
  2. Key T-cell marker genes are force-included regardless of Fano rank.
  3. Fano scores saved for downstream reference.

All other steps are identical to step3.py.

Input:  step2_cleaned.csv
Outputs:
  - step3_X_train.npy, step3_X_test.npy
  - step3_y_train.npy, step3_y_test.npy
  - step3_X_train_pca.npy, step3_X_test_pca.npy
  - step3_gene_names.npy
  - step3_gene_fano.npy          (NEW — Fano scores for selected genes)
  - step3_class_weights.npy
  - step3_label_mapping.json
  - step3_scaler.pkl
  - step3_pca.pkl
  - step3_config.json
  - step3_preprocessing_summary.png
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import json

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE  = "step2_cleaned.csv"
OUTPUT_DIR  = "."

N_TOP_GENES           = 3000
N_PCA_COMPONENTS      = 50
TEST_SIZE             = 0.2
RANDOM_STATE          = 42
MIN_CELLS_EXPRESSING  = 10

# Marker genes that must be in the feature set regardless of Fano rank
FORCED_MARKERS = [
    "CCR7", "SELL", "TCF7", "LEF1",             # Naive
    "GZMB", "GZMA", "PRF1", "NKG7", "GZMK",    # Effector
    "PDCD1", "HAVCR2", "LAG3", "TOX", "LAYN",  # Exhausted
    "FOXP3", "CTLA4", "IL2RA",                   # Treg
    "CXCL13", "IFNG", "BHLHE40",                # Th1-like
    "CD4", "CD8A", "CD8B",                       # General
]

# ============================================================
# STEP 3A: Load cleaned data
# ============================================================
print("=" * 60)
print("STEP 3A: Loading cleaned data from Step 2...")
print("=" * 60)

start = time.time()
df = pd.read_csv(INPUT_FILE)
print(f"  Loaded in {time.time()-start:.1f}s")
print(f"  Shape: {df.shape[0]} cells x {df.shape[1]} columns")

meta_cols = ["UniqueCell_ID", "Patient_ID", "majorCluster", "sampleType", "cell_class", "tissue"]
gene_cols  = [c for c in df.columns if c not in meta_cols]

X_raw      = df[gene_cols].values.astype(np.float32)
y_labels   = df["cell_class"].values
gene_names = np.array(gene_cols)

print(f"  Cells: {df.shape[0]}  Genes: {len(gene_cols)}  Classes: {df['cell_class'].nunique()}")
print(f"  Expression range: [{X_raw.min():.2f}, {X_raw.max():.2f}]")

# ============================================================
# STEP 3B: log2(TPM + 1) transformation
# ============================================================
print("\n" + "=" * 60)
print("STEP 3B: log2(TPM+1) transformation...")
print("=" * 60)

X_log = np.log2(X_raw + 1)
print(f"  Before: range [{X_raw.min():.1f}, {X_raw.max():.1f}]  mean={X_raw.mean():.2f}")
print(f"  After:  range [{X_log.min():.1f}, {X_log.max():.1f}]  mean={X_log.mean():.2f}")

# ============================================================
# STEP 3C: Filter low-expression genes
# ============================================================
print("\n" + "=" * 60)
print("STEP 3C: Filtering low-expression genes...")
print("=" * 60)

n_genes_before    = X_log.shape[1]
cells_expressing  = (X_raw > 0).sum(axis=0)
gene_mask         = cells_expressing >= MIN_CELLS_EXPRESSING
X_filtered        = X_log[:, gene_mask]
gene_names_filtered = gene_names[gene_mask]
n_genes_after     = X_filtered.shape[1]

print(f"  Threshold: expressed in ≥{MIN_CELLS_EXPRESSING} cells")
print(f"  Before: {n_genes_before}  After: {n_genes_after}  Removed: {n_genes_before-n_genes_after}")

# ============================================================
# STEP 3D: Select highly variable genes — FANO FACTOR
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 3D: Selecting top {N_TOP_GENES} HVGs by Fano factor (variance/mean)...")
print("=" * 60)

gene_variances = X_filtered.var(axis=0)
gene_means     = X_filtered.mean(axis=0)

# Fano factor = variance / mean.  Clip mean to avoid division by zero for
# essentially-silent genes (mean < 0.01 → Fano set to 0 so they rank last).
fano = np.where(gene_means > 0.01, gene_variances / (gene_means + 1e-8), 0.0)

top_gene_indices = np.argsort(fano)[::-1][:N_TOP_GENES]

# Force-include key T-cell marker genes regardless of Fano rank.
current_gene_set = set(gene_names_filtered[top_gene_indices])
missing_markers  = [g for g in FORCED_MARKERS
                    if g in gene_names_filtered and g not in current_gene_set]
if missing_markers:
    marker_indices   = np.where(np.isin(gene_names_filtered, missing_markers))[0]
    # Drop the lowest-ranked Fano genes to make room for the forced markers.
    top_gene_indices = np.concatenate([
        top_gene_indices[:N_TOP_GENES - len(marker_indices)],
        marker_indices
    ])
    print(f"  Force-included {len(missing_markers)} marker genes: {missing_markers}")
else:
    print(f"  All {len(FORCED_MARKERS)} marker genes already in top-{N_TOP_GENES} Fano selection.")

top_gene_indices = np.sort(top_gene_indices)

X_hvg          = X_filtered[:, top_gene_indices]
gene_names_hvg = gene_names_filtered[top_gene_indices]
fano_hvg       = fano[top_gene_indices]

print(f"  Selected {len(top_gene_indices)} genes")
print(f"  Fano range of selected: [{fano_hvg.min():.4f}, {fano_hvg.max():.4f}]")
print(f"  Feature matrix shape: {X_hvg.shape}")

# Report marker gene survival
print(f"\n  Key marker gene check:")
key_markers = {
    "Naive":     ["CCR7", "SELL", "TCF7", "LEF1"],
    "Effector":  ["GZMB", "GZMA", "PRF1", "NKG7"],
    "Exhausted": ["PDCD1", "HAVCR2", "LAG3", "TOX"],
    "Treg":      ["FOXP3", "CTLA4"],
    "General":   ["CD4", "CD8A", "CD8B", "IFNG", "CXCL13"],
}
for category, markers in key_markers.items():
    found   = [g for g in markers if g in gene_names_hvg]
    missing = [g for g in markers if g not in gene_names_hvg]
    status  = "✓" if not missing else "⚠"
    print(f"    {status} {category}: {found}" + (f"  (missing: {missing})" if missing else ""))

# ============================================================
# STEP 3E: Z-score normalisation (source only — per gene)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3E: Z-score normalisation (source)...")
print("=" * 60)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_hvg)
print(f"  After z-score: mean={X_scaled.mean():.6f}  std={X_scaled.std():.4f}")
print(f"  Shape: {X_scaled.shape}")

# ============================================================
# STEP 3F: Encode labels
# ============================================================
print("\n" + "=" * 60)
print("STEP 3F: Encoding class labels...")
print("=" * 60)

le        = LabelEncoder()
y_encoded = le.fit_transform(y_labels)
for i, cls in enumerate(le.classes_):
    print(f"    {i} → {cls} ({(y_encoded==i).sum()} cells)")

# ============================================================
# STEP 3G: Class weights
# ============================================================
print("\n" + "=" * 60)
print("STEP 3G: Class weights...")
print("=" * 60)

class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_encoded), y=y_encoded
)
class_weight_dict = {}
for i, (cls, w) in enumerate(zip(le.classes_, class_weights)):
    class_weight_dict[i] = w
    print(f"    {cls:<12}: {w:.3f}  (n={( y_encoded==i).sum()})")

# ============================================================
# STEP 3H: Train/test split (stratified)
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 3H: Train/test split (80/20, stratified)...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"  Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# ============================================================
# STEP 3I: PCA (for classical ML / visualisation)
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 3I: PCA ({N_PCA_COMPONENTS} components)...")
print("=" * 60)

pca          = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
X_train_pca  = pca.fit_transform(X_train)
X_test_pca   = pca.transform(X_test)
cum_var      = np.cumsum(pca.explained_variance_ratio_)
print(f"  {N_PCA_COMPONENTS} PCs explain {cum_var[-1]*100:.1f}% variance")

# ============================================================
# STEP 3J: Save all outputs
# ============================================================
print("\n" + "=" * 60)
print("STEP 3J: Saving outputs...")
print("=" * 60)

import pickle

np.save(os.path.join(OUTPUT_DIR, "step3_X_train.npy"),     X_train)
np.save(os.path.join(OUTPUT_DIR, "step3_X_test.npy"),      X_test)
np.save(os.path.join(OUTPUT_DIR, "step3_y_train.npy"),     y_train)
np.save(os.path.join(OUTPUT_DIR, "step3_y_test.npy"),      y_test)
np.save(os.path.join(OUTPUT_DIR, "step3_X_train_pca.npy"), X_train_pca)
np.save(os.path.join(OUTPUT_DIR, "step3_X_test_pca.npy"),  X_test_pca)
np.save(os.path.join(OUTPUT_DIR, "step3_gene_names.npy"),  gene_names_hvg)
np.save(os.path.join(OUTPUT_DIR, "step3_gene_fano.npy"),   fano_hvg)     # NEW
np.save(os.path.join(OUTPUT_DIR, "step3_class_weights.npy"), class_weights)

label_mapping = {int(i): str(cls) for i, cls in enumerate(le.classes_)}
with open(os.path.join(OUTPUT_DIR, "step3_label_mapping.json"), "w") as f:
    json.dump(label_mapping, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "step3_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(OUTPUT_DIR, "step3_pca.pkl"), "wb") as f:
    pickle.dump(pca, f)

config = {
    "n_top_genes":          N_TOP_GENES,
    "hvg_method":           "fano_factor",        # changed from raw variance
    "n_pca_components":     N_PCA_COMPONENTS,
    "test_size":            TEST_SIZE,
    "random_state":         RANDOM_STATE,
    "min_cells_expressing": MIN_CELLS_EXPRESSING,
    "n_cells_total":        len(y_encoded),
    "n_cells_train":        len(y_train),
    "n_cells_test":         len(y_test),
    "n_genes_original":     len(gene_cols),
    "n_genes_after_filter": n_genes_after,
    "n_genes_selected":     N_TOP_GENES,
    "forced_markers":       FORCED_MARKERS,
    "pca_variance_explained": float(cum_var[-1]),
    "label_mapping":        label_mapping,
    "class_weights":        {str(k): float(v) for k, v in class_weight_dict.items()},
}
with open(os.path.join(OUTPUT_DIR, "step3_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print("  Saved:")
for fname, shape in [
    ("step3_X_train.npy",     X_train.shape),
    ("step3_X_test.npy",      X_test.shape),
    ("step3_y_train.npy",     y_train.shape),
    ("step3_y_test.npy",      y_test.shape),
    ("step3_gene_names.npy",  gene_names_hvg.shape),
    ("step3_gene_fano.npy",   fano_hvg.shape),
]:
    print(f"    ✓ {fname}  {shape}")
print("    ✓ step3_scaler.pkl  step3_pca.pkl  step3_label_mapping.json  step3_config.json")

# ============================================================
# STEP 3K: Summary plots
# ============================================================
print("\n" + "=" * 60)
print("STEP 3K: Summary plots...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Fano factor distribution (all genes vs selected HVGs)
all_fano = fano[fano > 0]
axes[0, 0].hist(all_fano, bins=100, color="#95a5a6", alpha=0.7, label="All genes")
axes[0, 0].hist(fano_hvg[fano_hvg > 0], bins=100, color="#e74c3c", alpha=0.7,
                label=f"Top {N_TOP_GENES} HVGs")
axes[0, 0].set_title("Fano Factor Distribution (variance/mean)", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("Fano Factor"); axes[0, 0].set_ylabel("Number of Genes")
axes[0, 0].legend()

# Plot 2: PCA cumulative variance
axes[0, 1].plot(range(1, N_PCA_COMPONENTS+1), cum_var*100, "b-", linewidth=2)
axes[0, 1].axhline(90, color="r", linestyle="--", alpha=0.5, label="90% threshold")
n_for_90 = int(np.argmax(cum_var >= 0.9)) + 1
axes[0, 1].axvline(n_for_90, color="g", linestyle="--", alpha=0.5,
                   label=f"{n_for_90} PCs for 90%")
axes[0, 1].set_title("PCA Cumulative Variance Explained", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("Number of Components"); axes[0, 1].set_ylabel("Cumulative Variance (%)")
axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Expression distribution (most variable gene)
top_gene_name = gene_names_hvg[np.argmax(fano_hvg)]
top_gene_idx  = np.where(gene_names_hvg == top_gene_name)[0][0]
axes[1, 0].hist(X_hvg[:, top_gene_idx], bins=50, color="#e74c3c", alpha=0.8)
axes[1, 0].set_title(f"Highest Fano Gene: {top_gene_name}", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("log2(TPM+1)"); axes[1, 0].set_ylabel("Cell count")

# Plot 4: Class weights
class_label_names = [label_mapping[i] for i in range(len(label_mapping))]
bar_colors = ["#3498db", "#e74c3c", "#8e44ad", "#2ecc71", "#f39c12", "#95a5a6"]
axes[1, 1].bar(class_label_names, class_weights,
               color=bar_colors[:len(class_label_names)], edgecolor="white")
axes[1, 1].set_title("Class Weights (inverse frequency)", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Cell Class"); axes[1, 1].set_ylabel("Weight")
axes[1, 1].tick_params(axis="x", rotation=30)
axes[1, 1].axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="balanced=1.0")
axes[1, 1].legend()

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "step3_preprocessing_summary.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {plot_path}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 3 v2 COMPLETE")
print("=" * 60)
print(f"""
  Pipeline:
    1. log2(TPM+1) transform
    2. Low-expression filter      {n_genes_before} → {n_genes_after} genes
    3. Fano-factor HVG selection  {n_genes_after} → {N_TOP_GENES} genes  ← improved
    4. Forced marker genes        {len(FORCED_MARKERS)} guaranteed present
    5. Z-score normalisation      source only (mean=0, std=1 per gene)
    6. PCA ({N_PCA_COMPONENTS} components)      {cum_var[-1]*100:.1f}% variance
    7. Train/test split           {len(y_train)}/{len(y_test)} (stratified)

  Output for deep learning (full {N_TOP_GENES} genes):
    step3_X_train.npy  {X_train.shape}
    step3_X_test.npy   {X_test.shape}

  Next: run preprocess_target_v2.py to rebuild the target
  (gse126030_preprocessed_v2.npy) with independent standardisation.
""")
