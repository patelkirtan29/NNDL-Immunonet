"""
Label Validation & Re-clustering: GSE126030
==============================================
Problem: GSE126030 has condition-based labels (resting/activated) that don't 
map to our 6 T-cell classes (Naive, Effector, Exhausted, Treg, Th1-like, Other_CD4).

Solution: Re-cluster GSE126030 using the same marker genes as our training set,
then assign cells to the 6 classes based on similarity to class-specific expression patterns.

Output: 
  - gse126030_reclustered_labels.csv (cell_id, original_condition, new_class)
  - reclustering_report.json (statistics, confusion between old vs new labels)
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import glob
import pickle
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

GSE126030_DATA_PATH = PROJECT_ROOT / "gse126030_extracted"  # From cross_dataset.py
TRAINING_DATA_PATH = PROJECT_ROOT / "step3_X_train.npy"
TRAINING_LABELS_PATH = PROJECT_ROOT / "step3_y_train.npy"
TRAIN_GENE_NAMES_PATH = PROJECT_ROOT / "step3_gene_names.npy"
LABEL_MAP_PATH = PROJECT_ROOT / "step3_label_mapping.json"
PREPROCESSING_DIR = PROJECT_ROOT
OUTPUT_DIR = SCRIPT_DIR / "label_validation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Key marker genes for each class (from biological validation)
CLASS_MARKERS = {
    "Naive": ["CCR7", "SELL", "TCF7", "LEF1", "CD27", "CD28", "IL7R"],
    "Effector": ["GZMB", "GZMA", "PRF1", "NKG7", "IFNG", "GNLY"],
    "Exhausted": ["PDCD1", "HAVCR2", "LAG3", "TOX", "TIGIT", "LAYN", "CXCL13"],
    "Treg": ["FOXP3", "CTLA4", "IL2RA", "IL10", "TNFRSF18"],
    "Th1-like": ["CXCL13", "IFNG", "GZMK", "BHLHE40", "CD44"],
    "Other_CD4": ["CD4", "CD40LG", "ANXA1", "CXCR6"]
}

print("=" * 60)
print("LABEL VALIDATION & RE-CLUSTERING: GSE126030")
print("=" * 60)

# ============================================================
# STEP 1: Load training data to understand class signatures
# ============================================================
print("\nSTEP 1: Loading training data class signatures...")
print("-" * 60)

if not TRAINING_DATA_PATH.exists():
    print(f"  Error: Training feature matrix not found at {TRAINING_DATA_PATH}")
    print("  Run step3.py first to generate step3_X_train.npy")
    exit(1)

if not TRAINING_LABELS_PATH.exists():
    print(f"  Error: Training labels not found at {TRAINING_LABELS_PATH}")
    print("  Run step3.py first to generate step3_y_train.npy")
    exit(1)

if not LABEL_MAP_PATH.exists():
    print(f"  Error: Label mapping not found at {LABEL_MAP_PATH}")
    print("  Run step3.py first to generate step3_label_mapping.json")
    exit(1)

if not TRAIN_GENE_NAMES_PATH.exists():
    print(f"  Error: Training gene names not found at {TRAIN_GENE_NAMES_PATH}")
    print("  Run step3.py first to generate step3_gene_names.npy")
    exit(1)

X_train = np.load(TRAINING_DATA_PATH).astype(np.float32)
y_train_numeric = np.load(TRAINING_LABELS_PATH)
train_gene_names = np.load(TRAIN_GENE_NAMES_PATH, allow_pickle=True)
print(f"  Training data: {X_train.shape[0]} cells x {X_train.shape[1]} features")

# Load class weights and label mapping for reference
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
class_names_list = [label_map[str(i)] for i in range(len(label_map))]
print(f"  Classes: {class_names_list}")

# Compute mean expression per class
class_signatures = {}
for cls_idx, cls_name in enumerate(class_names_list):
    mask = (y_train_numeric == cls_idx)
    if mask.sum() > 0:
        mean_expr = X_train[mask].mean(axis=0)
        class_signatures[cls_name] = mean_expr
        print(f"  {cls_name}: {mask.sum()} cells, mean norm: {np.linalg.norm(mean_expr):.2f}")

# ============================================================
# STEP 2: Load preprocessed GSE126030 data
# ============================================================
print("\nSTEP 2: Loading GSE126030 preprocessed data...")
print("-" * 60)

# Load the preprocessed GSE126030 data (from cross_dataset.py output)
X_gse126030 = None
# For now, we'll load from a saved state in cross_dataset results

try:
    # This should have been saved during cross_dataset.py execution
    gse126030_data_path = PREPROCESSING_DIR / "gse126030_preprocessed.npy"
    if gse126030_data_path.exists():
        X_gse126030 = np.load(gse126030_data_path)
        print(f"  Loaded preprocessed GSE126030: {X_gse126030.shape}")
    else:
        print(f"  Warning: GSE126030 preprocessed data not found")
        print(f"  Expected at: {gse126030_data_path}")
        print(f"  Building it now from extracted GSE126030 matrices...")

        extracted_files = sorted(glob.glob(str(GSE126030_DATA_PATH / "*.gz")))
        if len(extracted_files) == 0:
            print(f"  Error: No .gz files found in {GSE126030_DATA_PATH}")
            print("  Ensure GSE126030 files are extracted before running this step")
            exit(1)

        train_gene_names_path = PREPROCESSING_DIR / "step3_gene_names.npy"
        scaler_path = PREPROCESSING_DIR / "step3_scaler.pkl"

        if not train_gene_names_path.exists():
            print(f"  Error: Missing training genes file at {train_gene_names_path}")
            exit(1)

        train_gene_names = np.load(train_gene_names_path, allow_pickle=True)

        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                training_scaler = pickle.load(f)
            print("  Loaded training scaler")
        else:
            training_scaler = None
            print("  Warning: step3_scaler.pkl missing, using log2-transformed values without scaling")

        matrices = []
        conditions = []

        for file_path in extracted_files:
            try:
                df_sample = pd.read_csv(file_path, sep="\t", compression="gzip")
                if "Gene" not in df_sample.columns or df_sample.shape[1] < 3:
                    continue

                sample_genes = df_sample["Gene"].astype(str).values
                sample_matrix = df_sample.iloc[:, 2:].values.T.astype(np.float32)

                aligned = np.zeros((sample_matrix.shape[0], len(train_gene_names)), dtype=np.float32)
                gene_to_idx = {g: i for i, g in enumerate(sample_genes)}
                for gene_idx, gene_name in enumerate(train_gene_names):
                    if gene_name in gene_to_idx:
                        aligned[:, gene_idx] = sample_matrix[:, gene_to_idx[gene_name]]

                matrices.append(aligned)

                filename_lower = Path(file_path).name.lower()
                if "activated" in filename_lower:
                    cond = "Activated"
                elif "resting" in filename_lower or "unstimulated" in filename_lower:
                    cond = "Resting"
                else:
                    cond = "Unknown"
                conditions.extend([cond] * aligned.shape[0])

            except Exception:
                continue

        if len(matrices) == 0:
            print("  Error: Could not parse any valid GSE126030 sample matrix")
            exit(1)

        X_raw = np.vstack(matrices).astype(np.float32)
        X_log = np.log2(X_raw + 1)
        if training_scaler is not None:
            X_gse126030 = training_scaler.transform(X_log)
        else:
            X_gse126030 = X_log
        X_gse126030 = np.nan_to_num(X_gse126030, nan=0.0)

        np.save(gse126030_data_path, X_gse126030)
        conditions_path = PREPROCESSING_DIR / "gse126030_original_conditions.npy"
        np.save(conditions_path, np.array(conditions, dtype=object))

        print(f"  Built preprocessed matrix: {X_gse126030.shape}")
        print(f"  Saved: {gse126030_data_path}")
        print(f"  Saved: {conditions_path}")
except Exception as e:
    print(f"  Error loading GSE126030 data: {e}")
    exit(1)

# ============================================================
# STEP 3: Build marker-based class scores
# ============================================================
print("\nSTEP 3: Scoring cells with class markers...")
print("-" * 60)

gene_to_idx = {gene: idx for idx, gene in enumerate(train_gene_names)}
class_marker_indices = {}
for cls_name, markers in CLASS_MARKERS.items():
    indices = [gene_to_idx[m] for m in markers if m in gene_to_idx]
    class_marker_indices[cls_name] = indices
    print(f"  {cls_name:<12}: {len(indices)} / {len(markers)} marker genes found")

score_matrix = np.zeros((X_gse126030.shape[0], len(class_names_list)), dtype=np.float32)
for class_idx, class_name in enumerate(class_names_list):
    indices = class_marker_indices.get(class_name, [])
    if len(indices) == 0:
        continue
    score_matrix[:, class_idx] = X_gse126030[:, indices].mean(axis=1)

score_matrix = StandardScaler().fit_transform(score_matrix)
print(f"  Marker-score matrix: {score_matrix.shape}")

# ============================================================
# STEP 4: Re-cluster cells in marker-score space
# ============================================================
print("\nSTEP 4: Re-clustering marker scores into 6 classes...")
print("-" * 60)

kmeans = KMeans(n_clusters=len(class_names_list), random_state=42, n_init=20)
cluster_ids = kmeans.fit_predict(score_matrix)

cluster_means = np.zeros((len(class_names_list), len(class_names_list)), dtype=np.float32)
for cluster_id in range(len(class_names_list)):
    mask = cluster_ids == cluster_id
    cluster_means[cluster_id] = score_matrix[mask].mean(axis=0) if mask.any() else 0

row_ind, col_ind = linear_sum_assignment(-cluster_means)
cluster_to_class = {int(cluster): class_names_list[int(cls_idx)] for cluster, cls_idx in zip(row_ind, col_ind)}

predicted_class_names = np.array([cluster_to_class[int(cluster_id)] for cluster_id in cluster_ids], dtype=object)

exp_scores = np.exp(score_matrix - score_matrix.max(axis=1, keepdims=True))
proba_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)
confidence = proba_scores.max(axis=1)

print(f"  Predicted labels for {len(predicted_class_names)} GSE126030 cells")
print(f"  Mean confidence: {confidence.mean():.3f} ± {confidence.std():.3f}")

unique_classes, counts = np.unique(predicted_class_names, return_counts=True)
print(f"\n  Predicted class distribution:")
for cls, count in zip(unique_classes, counts):
    pct = count / len(predicted_class_names) * 100
    print(f"    {cls:<12}: {count:>6} ({pct:.1f}%)")

# ============================================================
# STEP 5: Load original GSE126030 condition labels for comparison
# ============================================================
print("\nSTEP 5: Loading original GSE126030 condition labels...")
print("-" * 60)

# If we have the original condition labels from cross_dataset.py, load them
original_conditions = None
original_condition_path = PREPROCESSING_DIR / "gse126030_original_conditions.npy"
if original_condition_path.exists():
    original_conditions = np.load(original_condition_path, allow_pickle=True)
    print(f"  Loaded {len(original_conditions)} original condition labels")
    
    # Show relationship between original conditions and new predictions
    print(f"\n  Mapping: Original Condition → New Class Distribution")
    for cond in np.unique(original_conditions):
        mask = original_conditions == cond
        pred_dist = pd.Series(predicted_class_names[mask]).value_counts()
        print(f"\n    {cond} ({mask.sum()} cells):")
        for cls, count in pred_dist.items():
            pct = count / mask.sum() * 100
            print(f"      → {cls}: {count} ({pct:.1f}%)")
else:
    print(f"  Original condition labels not found")
    print(f"  Continuing with re-clustered labels only...")

# ============================================================
# STEP 6: Filter low-confidence predictions
# ============================================================
print("\nSTEP 6: Filtering low-confidence predictions...")
print("-" * 60)

CONFIDENCE_THRESHOLD = 0.35
high_confidence_mask = confidence >= CONFIDENCE_THRESHOLD
n_high_conf = high_confidence_mask.sum()
n_low_conf = (~high_confidence_mask).sum()

print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"  High confidence: {n_high_conf} cells ({n_high_conf/len(confidence)*100:.1f}%)")
print(f"  Low confidence: {n_low_conf} cells ({n_low_conf/len(confidence)*100:.1f}%)")

# Mark low-confidence cells
y_gse126030_final = predicted_class_names.astype(object).copy()
y_gse126030_final[~high_confidence_mask] = "Uncertain"

print(f"\n  Final class distribution (including Uncertain):")
final_dist = pd.Series(y_gse126030_final).value_counts()
for cls, count in final_dist.items():
    pct = count / len(y_gse126030_final) * 100
    print(f"    {cls:<12}: {count:>6} ({pct:.1f}%)")

# ============================================================
# STEP 7: Save re-clustered labels
# ============================================================
print("\nSTEP 7: Saving re-clustered labels...")
print("-" * 60)

# Create output dataframe
top3_classes = []
top3_scores = np.argsort(-proba_scores, axis=1)[:, :3]
for row in top3_scores:
    top3_classes.append(
        ", ".join([class_names_list[int(i)] for i in row])
    )

output_df = pd.DataFrame({
    "cell_id": np.arange(len(y_gse126030_final)),
    "original_condition": original_conditions if original_conditions is not None else ["Unknown"] * len(y_gse126030_final),
    "new_class": y_gse126030_final,
    "confidence": confidence,
    "top_k_distances": top3_classes
})

output_path = OUTPUT_DIR / "gse126030_reclustered_labels.csv"
output_df.to_csv(output_path, index=False)
print(f"  Saved: {output_path}")

# ============================================================
# STEP 8: Generate re-clustering report
# ============================================================
print("\nSTEP 8: Generating re-clustering report...")
print("-" * 60)

report = {
    "n_cells": len(y_gse126030_final),
    "n_features": X_gse126030.shape[1],
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "n_high_confidence": int(n_high_conf),
    "n_low_confidence": int(n_low_conf),
    "mean_confidence": float(confidence.mean()),
    "std_confidence": float(confidence.std()),
    "class_distribution": final_dist.to_dict(),
    "clustering_params": {
        "method": "kmeans_on_marker_scores",
        "n_clusters": len(class_names_list),
        "random_state": 42,
        "n_init": 20
    },
    "class_markers": {cls: markers for cls, markers in CLASS_MARKERS.items()},
    "cluster_to_class_mapping": cluster_to_class,
    "training_class_sizes": {cls_name: int(mask.sum()) 
                             for cls_idx, cls_name in enumerate(class_names_list)
                             for mask in [y_train_numeric == cls_idx]},
}

# Add condition → class mapping if available
if original_conditions is not None:
    condition_mapping = {}
    for cond in np.unique(original_conditions):
        mask = original_conditions == cond
        pred_dist = pd.Series(predicted_class_names[mask]).value_counts()
        condition_mapping[cond] = pred_dist.to_dict()
    report["condition_to_class_mapping"] = condition_mapping

report_path = os.path.join(OUTPUT_DIR, "reclustering_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"  Saved: {report_path}")

# ============================================================
# STEP 9: Visualize confidence distribution
# ============================================================
print("\nSTEP 9: Generating confidence visualization...")
print("-" * 60)

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of confidence scores
    axes[0].hist(confidence, bins=50, edgecolor='white', color='#3498db', alpha=0.7)
    axes[0].axvline(CONFIDENCE_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold ({CONFIDENCE_THRESHOLD})')
    axes[0].set_title('Distribution of Prediction Confidence', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Number of Cells')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Class distribution (stacked)
    class_counts = pd.Series(y_gse126030_final).value_counts()
    colors = ['#3498db', '#e74c3c', '#8e44ad', '#2ecc71', '#f39c12', '#95a5a6', '#95a5a6']
    axes[1].bar(class_counts.index, class_counts.values, color=colors[:len(class_counts)], edgecolor='white')
    axes[1].set_title('Re-clustered Class Distribution', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Cells')
    axes[1].tick_params(axis='x', rotation=30)
    
    for i, (cls, count) in enumerate(class_counts.items()):
        pct = count / len(y_gse126030_final) * 100
        axes[1].text(i, count + 500, f'{pct:.0f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    viz_path = OUTPUT_DIR / "reclustering_visualization.png"
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization: {viz_path}")
except ImportError:
    print("  Matplotlib not available, skipping visualization")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("LABEL VALIDATION COMPLETE")
print("=" * 60)
print(f"""
  Output files:
    1. {output_path}
       → GSE126030 cells with re-clustered labels
       → Includes confidence scores for filtering
    
    2. {report_path}
       → Statistics on re-clustering performance
       → Condition → class mapping

  Key findings:
    - {n_high_conf} cells ({n_high_conf/len(confidence)*100:.1f}%) have high confidence (≥{CONFIDENCE_THRESHOLD})
    - {n_low_conf} cells ({n_low_conf/len(confidence)*100:.1f}%) marked as "Uncertain"
    - Mean prediction confidence: {confidence.mean():.3f}

  Next step:
    Use the re-clustered labels with batch correction
    to fairly evaluate cross-dataset performance.
""")
