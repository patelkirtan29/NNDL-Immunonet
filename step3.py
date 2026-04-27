"""
ImmunoNet — Step 3: Preprocessing Pipeline
============================================
This script:
  1. Loads step2_cleaned.csv
  2. Applies log2(TPM+1) transformation
  3. Filters low-expression genes
  4. Selects highly variable genes (top 3,000)
  5. Z-score normalizes per gene
  6. Runs PCA for dimensionality reduction
  7. Computes class weights for imbalanced data
  8. Splits into train/test (80/20, stratified)
  9. Saves everything needed for model training

Requirements: pip install pandas numpy scikit-learn matplotlib seaborn

Input:  step2_cleaned.csv (from Step 2)
Outputs:
  - step3_X_train.npy, step3_X_test.npy       (feature matrices)
  - step3_y_train.npy, step3_y_test.npy        (label arrays)
  - step3_X_train_pca.npy, step3_X_test_pca.npy (PCA-reduced features)
  - step3_gene_names.npy                        (selected gene names)
  - step3_class_weights.npy                     (weights for imbalanced classes)
  - step3_label_encoder.npy                     (class name ↔ number mapping)
  - step3_preprocessing_summary.png             (visualization)
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
INPUT_FILE = "step2_cleaned.csv"
OUTPUT_DIR = "."

N_TOP_GENES = 3000           # Number of highly variable genes to keep
N_PCA_COMPONENTS = 50        # PCA components for reduced representation
TEST_SIZE = 0.2              # 80% train, 20% test
RANDOM_STATE = 42            # Reproducibility
MIN_CELLS_EXPRESSING = 10    # Gene must be expressed in at least this many cells

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

# Separate metadata and gene expression
meta_cols = ["UniqueCell_ID", "Patient_ID", "majorCluster", "sampleType", "cell_class", "tissue"]
gene_cols = [c for c in df.columns if c not in meta_cols]

print(f"  Cells: {df.shape[0]}")
print(f"  Genes: {len(gene_cols)}")
print(f"  Classes: {df['cell_class'].nunique()}")

# Extract expression matrix and labels
X_raw = df[gene_cols].values.astype(np.float32)
y_labels = df["cell_class"].values
gene_names = np.array(gene_cols)

print(f"  Expression matrix shape: {X_raw.shape}")
print(f"  Value range: [{X_raw.min():.2f}, {X_raw.max():.2f}]")

# ============================================================
# STEP 3B: Log2(TPM + 1) transformation
# ============================================================
print("\n" + "=" * 60)
print("STEP 3B: Applying log2(TPM + 1) transformation...")
print("=" * 60)

# log2(TPM+1) compresses the huge dynamic range of gene expression
# Raw TPM can range from 0 to 100,000+
# After log2: 0 stays 0, 1→1, 10→3.5, 100→6.7, 10000→13.3
# This makes the distribution more normal and easier for ML models

X_log = np.log2(X_raw + 1)

print(f"  Before log2: range [{X_raw.min():.1f}, {X_raw.max():.1f}], mean={X_raw.mean():.2f}")
print(f"  After log2:  range [{X_log.min():.1f}, {X_log.max():.1f}], mean={X_log.mean():.2f}")

# ============================================================
# STEP 3C: Filter low-expression genes
# ============================================================
print("\n" + "=" * 60)
print("STEP 3C: Filtering low-expression genes...")
print("=" * 60)

# Remove genes that are barely expressed — they add noise, not signal
# A gene must be expressed (TPM > 0) in at least MIN_CELLS_EXPRESSING cells
n_genes_before = X_log.shape[1]

cells_expressing = (X_raw > 0).sum(axis=0)  # count cells where gene > 0
gene_mask = cells_expressing >= MIN_CELLS_EXPRESSING

X_filtered = X_log[:, gene_mask]
gene_names_filtered = gene_names[gene_mask]

n_genes_after = X_filtered.shape[1]
n_removed = n_genes_before - n_genes_after

print(f"  Minimum cells expressing threshold: {MIN_CELLS_EXPRESSING}")
print(f"  Before: {n_genes_before} genes")
print(f"  Removed: {n_removed} low-expression genes")
print(f"  After: {n_genes_after} genes")

# ============================================================
# STEP 3D: Select highly variable genes (HVGs)
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 3D: Selecting top {N_TOP_GENES} highly variable genes...")
print("=" * 60)

# Highly variable genes have the most variation across cells
# These are the most informative for distinguishing cell types
# Genes with low variance look the same in every cell → useless for classification

gene_variances = X_filtered.var(axis=0)
gene_means = X_filtered.mean(axis=0)

# Select top N genes by variance
top_gene_indices = np.argsort(gene_variances)[::-1][:N_TOP_GENES]
top_gene_indices = np.sort(top_gene_indices)  # keep original order

X_hvg = X_filtered[:, top_gene_indices]
gene_names_hvg = gene_names_filtered[top_gene_indices]

print(f"  Selected {N_TOP_GENES} genes with highest variance")
print(f"  Variance range of selected: [{gene_variances[top_gene_indices].min():.4f}, {gene_variances[top_gene_indices].max():.4f}]")
print(f"  Feature matrix shape: {X_hvg.shape}")

# Check which key marker genes survived selection
key_markers = {
    "Naive": ["CCR7", "SELL", "TCF7", "LEF1"],
    "Effector": ["GZMB", "GZMA", "PRF1", "NKG7"],
    "Exhausted": ["PDCD1", "HAVCR2", "LAG3", "TOX"],
    "Treg": ["FOXP3", "CTLA4"],
    "General": ["CD4", "CD8A", "CD8B", "IFNG", "CXCL13"]
}

print(f"\n  Key marker genes retained after HVG selection:")
for category, markers in key_markers.items():
    found = [g for g in markers if g in gene_names_hvg]
    missing = [g for g in markers if g not in gene_names_hvg]
    status = "✓" if len(missing) == 0 else "⚠"
    print(f"    {status} {category}: {found}" + (f"  (missing: {missing})" if missing else ""))

# ============================================================
# STEP 3E: Z-score normalization
# ============================================================
print("\n" + "=" * 60)
print("STEP 3E: Z-score normalization (per gene)...")
print("=" * 60)

# Z-score: subtract mean, divide by std for each gene
# After this, each gene has mean=0, std=1
# This ensures no single gene dominates just because it's expressed at higher levels

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_hvg)

print(f"  After z-score: mean={X_scaled.mean():.6f}, std={X_scaled.std():.4f}")
print(f"  Shape: {X_scaled.shape}")

# ============================================================
# STEP 3F: Encode labels (string → number)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3F: Encoding class labels...")
print("=" * 60)

le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)

print(f"  Label mapping:")
for i, cls in enumerate(le.classes_):
    count = (y_encoded == i).sum()
    print(f"    {i} → {cls} ({count} cells)")

# ============================================================
# STEP 3G: Compute class weights for imbalanced data
# ============================================================
print("\n" + "=" * 60)
print("STEP 3G: Computing class weights for imbalanced data...")
print("=" * 60)

# Class weights inversely proportional to class frequency
# Rare classes get higher weight → model pays more attention to them
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded),
    y=y_encoded
)

class_weight_dict = {}
print(f"  Class weights (higher = rarer, gets more attention):")
for i, (cls, weight) in enumerate(zip(le.classes_, class_weights)):
    count = (y_encoded == i).sum()
    class_weight_dict[i] = weight
    print(f"    {cls:<12}: weight={weight:.3f}  (count={count})")

# ============================================================
# STEP 3H: Train/Test split (stratified by class)
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 3H: Train/Test split ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}, stratified)...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded  # ensures same class proportions in train and test
)

print(f"  Train set: {X_train.shape[0]} cells x {X_train.shape[1]} genes")
print(f"  Test set:  {X_test.shape[0]} cells x {X_test.shape[1]} genes")

print(f"\n  Class distribution check (train vs test):")
print(f"  {'Class':<12} {'Train':>6} {'Test':>6} {'Train%':>7} {'Test%':>7}")
print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
for i, cls in enumerate(le.classes_):
    n_train = (y_train == i).sum()
    n_test = (y_test == i).sum()
    pct_train = n_train / len(y_train) * 100
    pct_test = n_test / len(y_test) * 100
    print(f"  {cls:<12} {n_train:>6} {n_test:>6} {pct_train:>6.1f}% {pct_test:>6.1f}%")

# ============================================================
# STEP 3I: PCA dimensionality reduction
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 3I: PCA ({N_PCA_COMPONENTS} components)...")
print("=" * 60)

# PCA compresses 3,000 genes into 50 components that capture most variance
# Useful for: visualization, faster classical ML, reducing overfitting
# Note: DL models will use the full 3,000-gene features, not PCA

pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print(f"  Top 10 components explain: {cumulative_variance[9]*100:.1f}% of variance")
print(f"  Top 30 components explain: {cumulative_variance[29]*100:.1f}% of variance")
print(f"  All {N_PCA_COMPONENTS} components explain: {cumulative_variance[-1]*100:.1f}% of variance")
print(f"  PCA train shape: {X_train_pca.shape}")
print(f"  PCA test shape:  {X_test_pca.shape}")

# ============================================================
# STEP 3J: Save all outputs
# ============================================================
print("\n" + "=" * 60)
print("STEP 3J: Saving all outputs...")
print("=" * 60)

# Feature matrices (full 3,000 genes — for DL models)
np.save(os.path.join(OUTPUT_DIR, "step3_X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "step3_X_test.npy"), X_test)
print(f"  ✓ step3_X_train.npy  ({X_train.shape})")
print(f"  ✓ step3_X_test.npy   ({X_test.shape})")

# Labels
np.save(os.path.join(OUTPUT_DIR, "step3_y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "step3_y_test.npy"), y_test)
print(f"  ✓ step3_y_train.npy  ({y_train.shape})")
print(f"  ✓ step3_y_test.npy   ({y_test.shape})")

# PCA-reduced features (for classical ML and visualization)
np.save(os.path.join(OUTPUT_DIR, "step3_X_train_pca.npy"), X_train_pca)
np.save(os.path.join(OUTPUT_DIR, "step3_X_test_pca.npy"), X_test_pca)
print(f"  ✓ step3_X_train_pca.npy  ({X_train_pca.shape})")
print(f"  ✓ step3_X_test_pca.npy   ({X_test_pca.shape})")

# Gene names (which genes survived selection)
np.save(os.path.join(OUTPUT_DIR, "step3_gene_names.npy"), gene_names_hvg)
print(f"  ✓ step3_gene_names.npy   ({gene_names_hvg.shape})")

# Class weights
np.save(os.path.join(OUTPUT_DIR, "step3_class_weights.npy"), class_weights)
print(f"  ✓ step3_class_weights.npy ({class_weights.shape})")

# Label encoder mapping (save as JSON for easy reading)
label_mapping = {int(i): str(cls) for i, cls in enumerate(le.classes_)}
with open(os.path.join(OUTPUT_DIR, "step3_label_mapping.json"), "w") as f:
    json.dump(label_mapping, f, indent=2)
print(f"  ✓ step3_label_mapping.json")

# Save scaler and PCA for later use on new data
import pickle
with open(os.path.join(OUTPUT_DIR, "step3_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(OUTPUT_DIR, "step3_pca.pkl"), "wb") as f:
    pickle.dump(pca, f)
print(f"  ✓ step3_scaler.pkl")
print(f"  ✓ step3_pca.pkl")

# Save preprocessing config for reproducibility
config = {
    "n_top_genes": N_TOP_GENES,
    "n_pca_components": N_PCA_COMPONENTS,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "min_cells_expressing": MIN_CELLS_EXPRESSING,
    "n_cells_total": len(y_encoded),
    "n_cells_train": len(y_train),
    "n_cells_test": len(y_test),
    "n_genes_original": len(gene_cols),
    "n_genes_after_filter": n_genes_after,
    "n_genes_selected": N_TOP_GENES,
    "pca_variance_explained": float(cumulative_variance[-1]),
    "label_mapping": label_mapping,
    "class_weights": {str(k): float(v) for k, v in class_weight_dict.items()}
}
with open(os.path.join(OUTPUT_DIR, "step3_config.json"), "w") as f:
    json.dump(config, f, indent=2)
print(f"  ✓ step3_config.json")

# ============================================================
# STEP 3K: Generate preprocessing summary plots
# ============================================================
print("\n" + "=" * 60)
print("STEP 3K: Generating preprocessing summary plots...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Gene variance distribution (before and after HVG selection)
axes[0, 0].hist(gene_variances, bins=100, color="#95a5a6", alpha=0.7, label="All genes")
axes[0, 0].hist(gene_variances[top_gene_indices], bins=100, color="#e74c3c", alpha=0.7, label=f"Top {N_TOP_GENES} HVGs")
axes[0, 0].set_title("Gene Variance Distribution", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("Variance (log2 space)")
axes[0, 0].set_ylabel("Number of Genes")
axes[0, 0].legend()
axes[0, 0].set_xlim(0, gene_variances[top_gene_indices].max() * 1.1)

# Plot 2: PCA cumulative variance explained
axes[0, 1].plot(range(1, N_PCA_COMPONENTS + 1), cumulative_variance * 100, "b-", linewidth=2)
axes[0, 1].axhline(y=90, color="r", linestyle="--", alpha=0.5, label="90% threshold")
n_for_90 = np.argmax(cumulative_variance >= 0.9) + 1
axes[0, 1].axvline(x=n_for_90, color="g", linestyle="--", alpha=0.5, label=f"{n_for_90} PCs for 90%")
axes[0, 1].set_title("PCA Cumulative Variance Explained", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("Number of Components")
axes[0, 1].set_ylabel("Cumulative Variance (%)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Expression distribution before/after log transform (sample of genes)
sample_gene_idx = top_gene_indices[0]  # pick the most variable gene
sample_gene_name = gene_names_hvg[0]
axes[1, 0].hist(X_raw[:, gene_mask][:, top_gene_indices[0]], bins=50, color="#3498db", alpha=0.7, label="Raw TPM")
ax_twin = axes[1, 0].twinx()
ax_twin.hist(X_hvg[:, 0], bins=50, color="#e74c3c", alpha=0.5, label="log2(TPM+1)")
axes[1, 0].set_title(f"Expression Distribution: {sample_gene_name}", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Expression Value")
axes[1, 0].set_ylabel("Count (Raw TPM)", color="#3498db")
ax_twin.set_ylabel("Count (log2)", color="#e74c3c")
lines1, labels1 = axes[1, 0].get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
axes[1, 0].legend(lines1 + lines2, labels1 + labels2)

# Plot 4: Class weight visualization
class_names = [label_mapping[i] for i in range(len(label_mapping))]
bar_colors = ["#3498db", "#e74c3c", "#8e44ad", "#2ecc71", "#f39c12", "#95a5a6"]
axes[1, 1].bar(class_names, class_weights, color=bar_colors[:len(class_names)], edgecolor="white")
axes[1, 1].set_title("Class Weights (Inverse Frequency)", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Cell Class")
axes[1, 1].set_ylabel("Weight")
axes[1, 1].tick_params(axis="x", rotation=30)
axes[1, 1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Balanced = 1.0")
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
print("STEP 3 COMPLETE")
print("=" * 60)
print(f"""
  Preprocessing pipeline:
    1. log2(TPM+1) transform       — compressed dynamic range
    2. Low-expression filter        — {n_genes_before} → {n_genes_after} genes (removed {n_removed})
    3. Highly variable gene select  — {n_genes_after} → {N_TOP_GENES} genes
    4. Z-score normalization        — mean=0, std=1 per gene
    5. PCA                          — {N_TOP_GENES} → {N_PCA_COMPONENTS} components ({cumulative_variance[-1]*100:.1f}% variance)
    6. Train/test split             — {len(y_train)}/{len(y_test)} (stratified)

  Output files:
    For Deep Learning (full {N_TOP_GENES} genes):
      step3_X_train.npy       ({X_train.shape})
      step3_X_test.npy        ({X_test.shape})

    For Classical ML (PCA-reduced):
      step3_X_train_pca.npy   ({X_train_pca.shape})
      step3_X_test_pca.npy    ({X_test_pca.shape})

    Labels:
      step3_y_train.npy       ({y_train.shape})
      step3_y_test.npy        ({y_test.shape})

    Metadata:
      step3_gene_names.npy    — which {N_TOP_GENES} genes were selected
      step3_class_weights.npy — use in loss function for imbalanced classes
      step3_label_mapping.json — class number ↔ name mapping
      step3_scaler.pkl        — fitted scaler (reuse on new data)
      step3_pca.pkl           — fitted PCA (reuse on new data)
      step3_config.json       — all preprocessing parameters

  How to load in your training scripts:

    import numpy as np
    import json

    X_train = np.load("step3_X_train.npy")       # ({X_train.shape})
    X_test  = np.load("step3_X_test.npy")         # ({X_test.shape})
    y_train = np.load("step3_y_train.npy")        # ({y_train.shape})
    y_test  = np.load("step3_y_test.npy")         # ({y_test.shape})
    class_weights = np.load("step3_class_weights.npy")

    with open("step3_label_mapping.json") as f:
        label_map = json.load(f)
    # label_map = {{"0": "Effector", "1": "Exhausted", ...}}

  Next step: Run step4_eda.py for exploratory visualizations
    - UMAP colored by class
    - Key marker gene heatmap
    - Expression boxplots per class
""")