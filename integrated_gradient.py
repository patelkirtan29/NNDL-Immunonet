"""
ImmunoNet — Step 7: Integrated Gradients Interpretability
============================================================
This script:
  1. Loads trained MLP and 1D-CNN models from Step 6
  2. Computes Integrated Gradients for both models
  3. Identifies top genes per class driving predictions
  4. Validates against known T-cell biology markers
  5. Generates interpretability figures for your report

Integrated Gradients (IG) answers: "Which genes contributed most to
this prediction?" by computing the gradient of the output with respect
to each input gene, accumulated along a path from a baseline (zero
expression) to the actual input. Unlike SHAP which is model-agnostic,
IG is native to neural networks and uses TensorFlow's autodiff directly.

Requirements: pip install numpy tensorflow matplotlib seaborn

Input:  Trained models from results/mlp/ and results/cnn_1d/, step3_*.npy
Output: Interpretability figures and gene rankings in results/interpretability/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = "results/interpretability"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_IG_STEPS = 50         # Number of interpolation steps for IG (more = more accurate, slower)
N_TOP_GENES = 20        # Top genes to show per class
RANDOM_STATE = 42

# ============================================================
# LOAD DATA AND MODELS
# ============================================================
print("=" * 60)
print("Loading data and trained models...")
print("=" * 60)

X_test = np.load("step3_X_test.npy").astype(np.float32)
y_test = np.load("step3_y_test.npy")
gene_names = np.load("step3_gene_names.npy", allow_pickle=True)

with open("step3_label_mapping.json") as f:
    label_map = json.load(f)

class_names = [label_map[str(i)] for i in range(len(label_map))]
n_classes = len(class_names)
n_features = X_test.shape[1]

print(f"  Test set: {X_test.shape}")
print(f"  Classes: {class_names}")

# Load models
models = {}

mlp_path = "results/mlp/model.keras"
if os.path.exists(mlp_path):
    models["MLP"] = keras.models.load_model(mlp_path)
    print(f"  Loaded MLP model")
else:
    print(f"  MLP model not found at {mlp_path}")

cnn_path = "results/cnn_1d/model.keras"
if os.path.exists(cnn_path):
    models["1D-CNN"] = keras.models.load_model(cnn_path)
    print(f"  Loaded 1D-CNN model")
else:
    print(f"  1D-CNN model not found at {cnn_path}")

if len(models) == 0:
    print("  ERROR: No models found. Run Step 6 first.")
    exit()

# Known T-cell markers for biological validation
known_markers = {
    "Effector":  ["GZMB", "GZMA", "PRF1", "NKG7", "IFNG", "GNLY", "FGFBP2", "CX3CR1", "TBX21"],
    "Exhausted": ["PDCD1", "HAVCR2", "LAG3", "TOX", "LAYN", "ENTPD1", "TIGIT", "CXCL13", "CTLA4"],
    "Naive":     ["CCR7", "SELL", "TCF7", "LEF1", "IL7R", "FOXO1", "CD27", "CD28"],
    "Treg":      ["FOXP3", "CTLA4", "IL2RA", "IKZF2", "IL10", "TNFRSF18", "TIGIT"],
    "Th1-like":  ["CXCL13", "IFNG", "GZMK", "BHLHE40", "IL23R", "CCL20", "GZMB"],
    "Other_CD4": ["ANXA1", "GNLY", "CXCR6", "CXCR5"],
}


# ============================================================
# INTEGRATED GRADIENTS IMPLEMENTATION
# ============================================================
def compute_integrated_gradients(model, inputs, target_class, baseline=None, n_steps=50, is_cnn=False):
    """
    Compute Integrated Gradients for a batch of inputs.

    How it works:
    1. Start from a baseline (zero expression = "no signal")
    2. Create n_steps intermediate inputs between baseline and actual input
    3. For each intermediate, compute the gradient of the target class output
       with respect to the input genes
    4. Average all gradients and multiply by (input - baseline)

    Result: attribution score per gene per cell
    Higher score = that gene contributed more to predicting the target class

    Args:
        model: trained Keras model
        inputs: (batch, n_features) gene expression data
        target_class: which class to compute attributions for
        baseline: reference point (default: zero vector = no expression)
        n_steps: interpolation steps (more = more accurate)
        is_cnn: if True, reshape inputs for Conv1D
    Returns:
        attributions: (batch, n_features) importance per gene
    """

    if baseline is None:
        baseline = tf.zeros_like(inputs)

    # Generate interpolated inputs between baseline and actual input
    # alphas shape: (n_steps, 1, 1) — scales from 0 to 1
    alphas = tf.cast(tf.linspace(0.0, 1.0, n_steps + 1), dtype=tf.float32)

    # Accumulate gradients
    all_gradients = []

    for alpha in alphas:
        # Interpolated input: baseline + alpha * (input - baseline)
        interpolated = baseline + alpha * (inputs - baseline)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)

            if is_cnn:
                # Reshape for Conv1D: (batch, features, 1)
                interp_cnn = tf.reshape(interpolated, [-1, interpolated.shape[1], 1])
                predictions = model(interp_cnn, training=False)
            else:
                predictions = model(interpolated, training=False)

            # Get the prediction for the target class
            target_output = predictions[:, target_class]

        # Compute gradients of target class output w.r.t. input
        gradients = tape.gradient(target_output, interpolated)
        all_gradients.append(gradients)

    # Stack and average gradients
    all_gradients = tf.stack(all_gradients)  # (n_steps+1, batch, n_features)

    # Trapezoidal rule for integral approximation (more accurate than simple mean)
    avg_gradients = (all_gradients[:-1] + all_gradients[1:]) / 2.0
    avg_gradients = tf.reduce_mean(avg_gradients, axis=0)

    # Integrated Gradients = (input - baseline) * average_gradients
    attributions = (inputs - baseline) * avg_gradients

    return attributions.numpy()


# ============================================================
# COMPUTE IG FOR EACH MODEL
# ============================================================
all_model_results = {}

for model_name, model in models.items():
    print(f"\n{'=' * 60}")
    print(f"Computing Integrated Gradients for {model_name}...")
    print(f"{'=' * 60}")

    is_cnn = (model_name == "1D-CNN")
    start = time.time()

    # Compute IG for each class
    # For each class, we use cells that were predicted as that class
    y_pred = np.argmax(model.predict(
        X_test.reshape(-1, n_features, 1) if is_cnn else X_test,
        verbose=0
    ), axis=1)

    class_attributions = {}   # class_name → (n_cells_in_class, n_features)
    class_top_genes = {}      # class_name → top gene rankings

    for cls_idx, cls_name in enumerate(class_names):
        print(f"\n  Computing IG for class: {cls_name}...")

        # Get cells predicted as this class
        mask = y_pred == cls_idx
        n_cells = mask.sum()

        if n_cells == 0:
            print(f"    No cells predicted as {cls_name}, skipping")
            continue

        # Use up to 200 cells per class (for speed)
        max_cells = min(200, n_cells)
        cell_indices = np.where(mask)[0][:max_cells]
        X_subset = tf.constant(X_test[cell_indices])

        # Compute IG
        attributions = compute_integrated_gradients(
            model=model,
            inputs=X_subset,
            target_class=cls_idx,
            n_steps=N_IG_STEPS,
            is_cnn=is_cnn
        )

        # Average absolute attributions across cells
        mean_attr = np.abs(attributions).mean(axis=0)

        # Get top genes
        top_idx = np.argsort(mean_attr)[::-1][:N_TOP_GENES]
        top_genes = gene_names[top_idx]
        top_scores = mean_attr[top_idx]

        class_attributions[cls_name] = mean_attr
        class_top_genes[cls_name] = {
            "genes": top_genes.tolist(),
            "scores": top_scores.tolist()
        }

        print(f"    Used {max_cells} cells, Top 5: {top_genes[:5].tolist()}")

    elapsed = time.time() - start
    print(f"\n  {model_name} IG completed in {elapsed:.1f}s")

    all_model_results[model_name] = {
        "attributions": class_attributions,
        "top_genes": class_top_genes
    }

    # ============================================================
    # FIGURE: Top Genes per Class (IG Attribution)
    # ============================================================
    print(f"\n  Generating figures for {model_name}...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, cls_name in enumerate(class_names):
        if idx >= 6 or cls_name not in class_top_genes:
            if idx < 6:
                axes[idx].set_visible(False)
            continue

        genes = class_top_genes[cls_name]["genes"][:15]
        scores = class_top_genes[cls_name]["scores"][:15]

        # Color known markers differently
        expected = known_markers.get(cls_name, [])
        bar_colors = ["#e74c3c" if g in expected else "#3498db" for g in genes]

        axes[idx].barh(range(len(genes)), scores[::-1], color=bar_colors[::-1], edgecolor="white")
        axes[idx].set_yticks(range(len(genes)))
        axes[idx].set_yticklabels(genes[::-1], fontsize=9)
        axes[idx].set_title(f"{cls_name}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("IG Attribution Score", fontsize=9)

        # Count known markers found
        found = [g for g in genes if g in expected]
        if found:
            axes[idx].set_title(f"{cls_name} (✓ {len(found)} known markers)", fontsize=10, fontweight="bold")

    for idx in range(len(class_names), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Top 15 Genes per Class — Integrated Gradients ({model_name})\n(Red = known T-cell marker, Blue = model-discovered)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"ig_top_genes_{model_name.lower().replace(' ', '_').replace('-', '_')}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# BIOLOGICAL VALIDATION: Overlap with Known Markers
# ============================================================
print(f"\n{'=' * 60}")
print("BIOLOGICAL VALIDATION: Known Marker Overlap")
print(f"{'=' * 60}")

validation_results = {}

for model_name, results in all_model_results.items():
    print(f"\n  {model_name}:")
    print(f"  {'Class':<12} {'Top 20 found':>12} {'Known markers':>14} {'Overlap':>8} {'Overlap %':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*8} {'-'*10}")

    model_validation = {}
    total_found = 0
    total_expected = 0

    for cls_name in class_names:
        if cls_name not in results["top_genes"]:
            continue

        top_20 = results["top_genes"][cls_name]["genes"][:20]
        expected = known_markers.get(cls_name, [])
        expected_in_data = [g for g in expected if g in gene_names]

        overlap = [g for g in top_20 if g in expected_in_data]
        n_overlap = len(overlap)
        n_expected = len(expected_in_data)
        pct = (n_overlap / n_expected * 100) if n_expected > 0 else 0

        total_found += n_overlap
        total_expected += n_expected

        model_validation[cls_name] = {
            "top_20_genes": top_20,
            "known_markers_in_data": expected_in_data,
            "overlap": overlap,
            "overlap_pct": pct
        }

        status = "✅" if pct >= 50 else "⚠️"
        print(f"  {cls_name:<12} {len(top_20):>12} {n_expected:>14} {n_overlap:>8} {pct:>9.0f}% {status}")

    overall_pct = (total_found / total_expected * 100) if total_expected > 0 else 0
    print(f"\n  Overall overlap: {total_found}/{total_expected} ({overall_pct:.0f}%)")
    target_met = overall_pct >= 70
    print(f"  Target (≥70%): {'✅ PASS' if target_met else '⚠️ BELOW — check gene rankings'}")

    validation_results[model_name] = model_validation


# ============================================================
# FIGURE: Marker Overlap Heatmap
# ============================================================
print(f"\n{'=' * 60}")
print("Generating marker overlap heatmap...")
print(f"{'=' * 60}")

for model_name, results in all_model_results.items():
    # Build matrix: rows = classes, columns = known markers
    all_known_markers = []
    marker_to_class = {}
    for cls, markers in known_markers.items():
        for m in markers:
            if m in gene_names and m not in all_known_markers:
                all_known_markers.append(m)
                marker_to_class[m] = cls

    if len(all_known_markers) == 0:
        continue

    # Attribution score for each marker in each class
    heatmap_data = np.zeros((n_classes, len(all_known_markers)))

    for i, cls_name in enumerate(class_names):
        if cls_name in results["attributions"]:
            attrs = results["attributions"][cls_name]
            for j, marker in enumerate(all_known_markers):
                if marker in gene_names:
                    gene_idx = np.where(gene_names == marker)[0][0]
                    heatmap_data[i, j] = attrs[gene_idx]

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        heatmap_data,
        xticklabels=all_known_markers,
        yticklabels=class_names,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "IG Attribution Score"}
    )

    ax.set_title(f"Integrated Gradients Attribution for Known T-Cell Markers — {model_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Known Marker Gene", fontsize=11)
    ax.set_ylabel("Cell Class", fontsize=11)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"ig_marker_heatmap_{model_name.lower().replace(' ', '_').replace('-', '_')}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# FIGURE: Compare Top Genes Across Models
# ============================================================
if len(all_model_results) > 1:
    print(f"\n{'=' * 60}")
    print("Comparing gene importance across models...")
    print(f"{'=' * 60}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, cls_name in enumerate(class_names):
        if idx >= 6:
            break

        ax = axes[idx]

        # Collect top 10 genes from each model
        model_genes = {}
        for model_name in all_model_results:
            if cls_name in all_model_results[model_name]["top_genes"]:
                model_genes[model_name] = all_model_results[model_name]["top_genes"][cls_name]["genes"][:10]

        if not model_genes:
            ax.set_visible(False)
            continue

        # Find genes that appear in multiple models (consensus)
        all_genes = []
        for genes in model_genes.values():
            all_genes.extend(genes)

        gene_counts = pd.Series(all_genes).value_counts()
        consensus_genes = gene_counts[gene_counts > 1].index.tolist()

        # Plot
        y_pos = 0
        colors_map = {"MLP": "#e74c3c", "1D-CNN": "#3498db"}
        for model_name, genes in model_genes.items():
            for gene in genes[:8]:
                color = colors_map.get(model_name, "#95a5a6")
                if gene in consensus_genes:
                    color = "#2ecc71"  # green for consensus
                ax.barh(y_pos, 1, color=color, edgecolor="white", height=0.8)
                ax.text(0.05, y_pos, f"{gene} ({model_name})", va="center", fontsize=8)
                y_pos += 1
            y_pos += 0.5  # gap between models

        ax.set_title(f"{cls_name}", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1.2)
        ax.set_yticks([])
        ax.set_xticks([])

    for idx in range(len(class_names), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Gene Importance Comparison Across Models\n(Green = found by multiple models)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ig_model_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# SAVE ALL RESULTS
# ============================================================
print(f"\n{'=' * 60}")
print("Saving all interpretability results...")
print(f"{'=' * 60}")

# Save top genes per model per class
save_data = {}
for model_name, results in all_model_results.items():
    save_data[model_name] = results["top_genes"]

with open(os.path.join(OUTPUT_DIR, "ig_top_genes_all_models.json"), "w") as f:
    json.dump(save_data, f, indent=2)

# Save validation results
with open(os.path.join(OUTPUT_DIR, "biological_validation.json"), "w") as f:
    json.dump(validation_results, f, indent=2, default=str)

print(f"  Saved: ig_top_genes_all_models.json")
print(f"  Saved: biological_validation.json")


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'=' * 60}")
print("STEP 7 COMPLETE — Integrated Gradients Interpretability")
print(f"{'=' * 60}")
print(f"""
  What we computed:
    Integrated Gradients (IG) attribution scores for each gene,
    for each class, for each DL model. IG tells us exactly which
    genes pushed the model toward each prediction.

  Biological validation:
    We checked whether the top IG-ranked genes match known T-cell
    markers from immunology literature. For example:
      - Does the model focus on GZMB/PRF1 for Effector? (killing genes)
      - Does it focus on PDCD1/TOX for Exhausted? (exhaustion genes)
      - Does it focus on FOXP3/CTLA4 for Treg? (regulatory genes)

  Files in {OUTPUT_DIR}/:
    ig_top_genes_mlp.png              — Top 15 genes per class (MLP)
    ig_top_genes_1d_cnn.png           — Top 15 genes per class (1D-CNN)
    ig_marker_heatmap_mlp.png         — Known markers attribution heatmap (MLP)
    ig_marker_heatmap_1d_cnn.png      — Known markers attribution heatmap (1D-CNN)
    ig_model_comparison.png           — Cross-model gene comparison
    ig_top_genes_all_models.json      — All gene rankings
    biological_validation.json        — Marker overlap statistics

  Key figures for your report:
    - ig_marker_heatmap → shows the model learned real biology
    - ig_top_genes → which genes drive each class prediction
    - biological_validation.json → overlap % for your paper's table

  Next step: Step 8 (Cross-dataset validation) or Step 9 (Final comparison)
""")