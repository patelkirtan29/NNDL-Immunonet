"""
ImmunoNet — Step 4: Exploratory Data Analysis & Visualization
===============================================================
This script:
  1. Generates UMAP embedding colored by cell class
  2. Creates marker gene expression heatmap
  3. Plots per-class expression boxplots for key genes
  4. Shows correlation between marker genes
  5. Visualizes data distribution with t-SNE
  All figures are publication-quality for your report and presentation.

Requirements: pip install numpy matplotlib seaborn scikit-learn umap-learn

Input:  step3_*.npy files (from Step 3), step2_cleaned.csv (for metadata)
Output: Multiple PNG figures in figures/ directory
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import json
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette for 6 classes — consistent across all plots
CLASS_COLORS = {
    "Effector":  "#e74c3c",
    "Exhausted": "#8e44ad",
    "Naive":     "#3498db",
    "Other_CD4": "#95a5a6",
    "Th1-like":  "#f39c12",
    "Treg":      "#2ecc71"
}

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("Loading preprocessed data...")
print("=" * 60)

X_train = np.load("step3_X_train.npy")
X_test = np.load("step3_X_test.npy")
y_train = np.load("step3_y_train.npy")
y_test = np.load("step3_y_test.npy")
gene_names = np.load("step3_gene_names.npy", allow_pickle=True)

with open("step3_label_mapping.json") as f:
    label_map = json.load(f)

# Combine train+test for EDA (we want to visualize all data)
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

# Convert numeric labels back to class names
y_names = np.array([label_map[str(i)] for i in y_all])

print(f"  Total cells: {X_all.shape[0]}")
print(f"  Genes: {X_all.shape[1]}")
print(f"  Classes: {len(label_map)}")

# ============================================================
# FIGURE 1: UMAP Embedding
# ============================================================
print("\n" + "=" * 60)
print("FIGURE 1: UMAP embedding (this takes 1-2 minutes)...")
print("=" * 60)

try:
    import umap

    start = time.time()

    # First reduce to 50 PCA components, then UMAP to 2D
    # This is the standard approach — UMAP on raw high-dim is slow
    pca_for_umap = PCA(n_components=50, random_state=42)
    X_pca = pca_for_umap.fit_transform(X_all)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.3,
        metric="euclidean",
        random_state=42
    )
    X_umap = reducer.fit_transform(X_pca)

    print(f"  UMAP computed in {time.time()-start:.1f}s")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    for cls in CLASS_COLORS:
        mask = y_names == cls
        ax.scatter(
            X_umap[mask, 0], X_umap[mask, 1],
            c=CLASS_COLORS[cls], label=cls,
            s=8, alpha=0.5, edgecolors="none"
        )

    ax.set_title("UMAP Embedding of T-Cell Gene Expression", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.legend(markerscale=3, fontsize=10, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_umap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Save UMAP coordinates for later use
    np.save(os.path.join(OUTPUT_DIR, "umap_coordinates.npy"), X_umap)

except ImportError:
    print("  umap-learn not installed. Run: pip install umap-learn")
    print("  Skipping UMAP, generating PCA plot instead...")

    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in CLASS_COLORS:
        mask = y_names == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=CLASS_COLORS[cls], label=cls, s=8, alpha=0.5, edgecolors="none")

    ax.set_title("PCA Projection of T-Cell Gene Expression", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.legend(markerscale=3, fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_pca_2d.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ============================================================
# FIGURE 2: Key Marker Gene Heatmap
# ============================================================
print("\n" + "=" * 60)
print("FIGURE 2: Marker gene expression heatmap...")
print("=" * 60)

# Define key markers per class
marker_genes = {
    "Naive":     ["CCR7", "SELL", "TCF7", "LEF1"],
    "Effector":  ["GZMB", "GZMA", "PRF1", "NKG7"],
    "Exhausted": ["PDCD1", "HAVCR2", "LAG3", "TOX", "LAYN"],
    "Treg":      ["FOXP3", "CTLA4", "IL2RA"],
    "Th1-like":  ["CXCL13", "IFNG", "GZMK"],
    "General":   ["CD4", "CD8A", "CD8B", "MKI67"]
}

# Find which markers exist in our selected genes
all_markers = []
marker_labels = []
for category, genes in marker_genes.items():
    for g in genes:
        if g in gene_names:
            all_markers.append(g)
            marker_labels.append(category)

print(f"  Found {len(all_markers)}/{sum(len(v) for v in marker_genes.values())} marker genes in selected features")

if len(all_markers) > 0:
    # Get expression values for marker genes
    marker_indices = [np.where(gene_names == g)[0][0] for g in all_markers]

    # Compute mean expression per class per marker gene
    heatmap_data = np.zeros((len(label_map), len(all_markers)))
    class_names_ordered = [label_map[str(i)] for i in range(len(label_map))]

    for i in range(len(label_map)):
        mask = y_all == i
        if mask.sum() > 0:
            heatmap_data[i, :] = X_all[mask][:, marker_indices].mean(axis=0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        heatmap_data,
        xticklabels=all_markers,
        yticklabels=class_names_ordered,
        cmap="RdYlBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Mean Z-scored Expression"}
    )

    ax.set_title("Mean Expression of Key Marker Genes per T-Cell Class", fontsize=13, fontweight="bold")
    ax.set_xlabel("Marker Gene", fontsize=11)
    ax.set_ylabel("Cell Class", fontsize=11)
    ax.tick_params(axis="x", rotation=45)

    # Add category labels on top
    prev_cat = ""
    for idx, (gene, cat) in enumerate(zip(all_markers, marker_labels)):
        if cat != prev_cat:
            ax.text(idx + 0.5, -0.8, cat, ha="center", fontsize=8, fontstyle="italic", color="gray")
            prev_cat = cat

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_marker_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ============================================================
# FIGURE 3: Expression Boxplots for Key Genes
# ============================================================
print("\n" + "=" * 60)
print("FIGURE 3: Expression boxplots for key genes...")
print("=" * 60)

# One representative gene per class
representative_genes = {
    "CCR7":  "Naive marker",
    "GZMB":  "Effector marker",
    "PDCD1": "Exhaustion marker (PD-1)",
    "FOXP3": "Treg marker",
    "CXCL13": "Th1-like marker",
    "CD8A":  "CD8+ T-cell marker"
}

# Filter to genes that exist in our data
plot_genes = {g: desc for g, desc in representative_genes.items() if g in gene_names}

if len(plot_genes) > 0:
    n_genes = len(plot_genes)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (gene, description) in enumerate(plot_genes.items()):
        if idx >= 6:
            break

        gene_idx = np.where(gene_names == gene)[0][0]
        gene_expr = X_all[:, gene_idx]

        # Build dataframe for seaborn
        plot_df = pd.DataFrame({
            "Expression": gene_expr,
            "Class": y_names
        })

        # Order classes consistently
        class_order = ["Naive", "Effector", "Exhausted", "Treg", "Th1-like", "Other_CD4"]
        class_order = [c for c in class_order if c in plot_df["Class"].unique()]

        palette = [CLASS_COLORS[c] for c in class_order]

        sns.boxplot(
            data=plot_df, x="Class", y="Expression",
            order=class_order, palette=palette,
            fliersize=1, linewidth=0.8,
            ax=axes[idx]
        )

        axes[idx].set_title(f"{gene} — {description}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("Z-scored Expression", fontsize=9)
        axes[idx].tick_params(axis="x", rotation=30, labelsize=9)

    # Hide unused axes
    for idx in range(len(plot_genes), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Expression of Key Marker Genes Across T-Cell Classes",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_gene_boxplots.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ============================================================
# FIGURE 4: Marker Gene Correlation Matrix
# ============================================================
print("\n" + "=" * 60)
print("FIGURE 4: Marker gene correlation matrix...")
print("=" * 60)

if len(all_markers) > 0:
    marker_expr = X_all[:, marker_indices]
    corr_matrix = np.corrcoef(marker_expr.T)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        xticklabels=all_markers,
        yticklabels=all_markers,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        square=True,
        cbar_kws={"label": "Pearson Correlation"}
    )

    ax.set_title("Correlation Between Key Marker Genes", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_gene_correlation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ============================================================
# FIGURE 5: Class Distribution in Train vs Test
# ============================================================
print("\n" + "=" * 60)
print("FIGURE 5: Train vs Test class distribution...")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

class_order = ["Naive", "Effector", "Exhausted", "Treg", "Th1-like", "Other_CD4"]

# Train set
y_train_names = [label_map[str(i)] for i in y_train]
train_counts = pd.Series(y_train_names).value_counts()
train_ordered = [train_counts.get(c, 0) for c in class_order]
colors_ordered = [CLASS_COLORS[c] for c in class_order]

axes[0].bar(class_order, train_ordered, color=colors_ordered, edgecolor="white")
axes[0].set_title(f"Training Set ({len(y_train)} cells)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=30)
for i, count in enumerate(train_ordered):
    axes[0].text(i, count + 20, str(count), ha="center", fontsize=9)

# Test set
y_test_names = [label_map[str(i)] for i in y_test]
test_counts = pd.Series(y_test_names).value_counts()
test_ordered = [test_counts.get(c, 0) for c in class_order]

axes[1].bar(class_order, test_ordered, color=colors_ordered, edgecolor="white")
axes[1].set_title(f"Test Set ({len(y_test)} cells)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=30)
for i, count in enumerate(test_ordered):
    axes[1].text(i, count + 5, str(count), ha="center", fontsize=9)

plt.suptitle("Stratified Train/Test Split — Class Distribution Preserved",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig5_train_test_split.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")

# ============================================================
# FIGURE 6: Top Genes by Variance
# ============================================================
print("\n" + "=" * 60)
print("FIGURE 6: Top 20 most variable genes...")
print("=" * 60)

gene_var = X_all.var(axis=0)
top20_idx = np.argsort(gene_var)[::-1][:20]
top20_names = gene_names[top20_idx]
top20_vars = gene_var[top20_idx]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(range(20), top20_vars[::-1], color="#3498db", edgecolor="white")
ax.set_yticks(range(20))
ax.set_yticklabels(top20_names[::-1], fontsize=10)
ax.set_xlabel("Variance (Z-scored)", fontsize=11)
ax.set_title("Top 20 Most Variable Genes (Highest Information Content)", fontsize=13, fontweight="bold")

# Highlight known marker genes
known_markers_flat = [g for genes in marker_genes.values() for g in genes]
for i, name in enumerate(top20_names[::-1]):
    if name in known_markers_flat:
        bars[i].set_color("#e74c3c")
        ax.text(top20_vars[::-1][i] + 0.02, i, "★ known marker", fontsize=8, va="center", color="#e74c3c")

ax.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, fc="#3498db", label="Other genes"),
        plt.Rectangle((0, 0), 1, 1, fc="#e74c3c", label="Known T-cell markers")
    ],
    fontsize=9
)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig6_top_variable_genes.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 4 COMPLETE — All EDA Figures Generated")
print("=" * 60)
print(f"""
  Figures saved in '{OUTPUT_DIR}/' directory:

    fig1_umap.png              — UMAP embedding showing cell clusters
    fig2_marker_heatmap.png    — Mean expression of markers per class
    fig3_gene_boxplots.png     — Expression distribution per class for key genes
    fig4_gene_correlation.png  — Correlation between marker genes
    fig5_train_test_split.png  — Class distribution preserved in train/test
    fig6_top_variable_genes.png — Most informative genes by variance

  Use these figures in:
    - Your 3/16 presentation (slides 1-2 for motivation)
    - Your final report (Results section)
    - Your paper (figures demonstrating data characteristics)

  Key things to look for in the plots:
    - UMAP: Do the 6 classes form distinct clusters? (they should)
    - Heatmap: Is GZMB high in Effector but low in Naive? (validates biology)
    - Boxplots: Is PDCD1 specifically high in Exhausted? (validates labels)
    - Correlation: Are exhaustion markers (PDCD1, HAVCR2, LAG3) correlated? (they should be)

  Next step: Run step5_classical_baselines.py to train SVM, RF, XGBoost
""")