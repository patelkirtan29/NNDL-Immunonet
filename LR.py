"""
ImmunoNet — Step 5a: Logistic Regression Baseline
===================================================
This script:
  1. Loads preprocessed data from Step 3
  2. Trains Logistic Regression (elastic net) with stratified 5-fold CV
  3. Tunes hyperparameters with RandomizedSearchCV
  4. Evaluates on held-out test set
  5. Generates confusion matrix, ROC curves, classification report
  6. Runs quick SHAP feature importance (top genes)

This is your FIRST model — run this to verify the entire pipeline
works before moving to SVM, RF, XGBoost, and deep learning.

Requirements: pip install numpy scikit-learn matplotlib seaborn shap

Input:  step3_*.npy files (from Step 3)
Output: Results and figures in results/logistic_regression/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = "results/logistic_regression"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_CV_FOLDS = 5
N_SEARCH_ITER = 30  # hyperparameter search iterations

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
class_weights = np.load("step3_class_weights.npy")

with open("step3_label_mapping.json") as f:
    label_map = json.load(f)

class_names = [label_map[str(i)] for i in range(len(label_map))]
n_classes = len(class_names)

print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  Classes: {class_names}")

# ============================================================
# STEP 5A-1: Hyperparameter Tuning with RandomizedSearchCV
# ============================================================
print("\n" + "=" * 60)
print("STEP 5A-1: Hyperparameter tuning (RandomizedSearchCV)...")
print(f"  {N_SEARCH_ITER} random combinations x {N_CV_FOLDS}-fold CV")
print("=" * 60)

# Hyperparameter search space
# param_dist = {
#     "C": [0.001, 0.01, 0.1, 1, 10, 100],           # Regularization strength (lower = more regularization)
#     "penalty": ["l1", "l2", "elasticnet"],            # Regularization type
#     "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],          # Only used for elasticnet
#     "solver": ["saga"],                                # Supports all penalties + large datasets
#     "max_iter": [500],                                # Enough iterations to converge
#     "class_weight": ["balanced"],                      # Handle imbalanced classes
# }

param_dist = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],           # Regularization strength (lower = more regularization)
    "penalty": ["l2"],            # Regularization type
    # "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],          # Only used for elasticnet
    "solver": ["lbfgs"],                                # Supports all penalties + large datasets
    "max_iter": [2000],                                # Enough iterations to converge
    "class_weight": ["balanced"],                      # Handle imbalanced classes
}

# Base model
lr_base = LogisticRegression(
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Stratified K-Fold
cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Search
start = time.time()
search = RandomizedSearchCV(
    estimator=lr_base,
    param_distributions=param_dist,
    n_iter=N_SEARCH_ITER,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)
search.fit(X_train, y_train)

print(f"\n  Search completed in {time.time()-start:.1f}s")
print(f"  Best macro-F1 (CV): {search.best_score_:.4f}")
print(f"  Best parameters:")
for param, value in search.best_params_.items():
    print(f"    {param}: {value}")

best_model = search.best_estimator_

# ============================================================
# STEP 5A-2: Cross-Validation Results (per-fold)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5A-2: 5-Fold Cross-Validation Results...")
print("=" * 60)

# Get per-fold scores with best model
fold_scores = []
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    # Clone and fit
    fold_model = LogisticRegression(**search.best_params_,
                                     random_state=RANDOM_STATE,
                                     n_jobs=-1)
    fold_model.fit(X_fold_train, y_fold_train)
    y_fold_pred = fold_model.predict(X_fold_val)

    fold_f1 = f1_score(y_fold_val, y_fold_pred, average="macro")
    fold_acc = accuracy_score(y_fold_val, y_fold_pred)
    fold_scores.append({"fold": fold_idx + 1, "macro_f1": fold_f1, "accuracy": fold_acc})

    print(f"  Fold {fold_idx+1}: macro-F1={fold_f1:.4f}, accuracy={fold_acc:.4f}")

mean_f1 = np.mean([s["macro_f1"] for s in fold_scores])
std_f1 = np.std([s["macro_f1"] for s in fold_scores])
mean_acc = np.mean([s["accuracy"] for s in fold_scores])
std_acc = np.std([s["accuracy"] for s in fold_scores])

print(f"\n  CV macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"  CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

# ============================================================
# STEP 5A-3: Evaluate on Held-Out Test Set
# ============================================================
print("\n" + "=" * 60)
print("STEP 5A-3: Test set evaluation...")
print("=" * 60)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

test_f1_macro = f1_score(y_test, y_pred, average="macro")
test_f1_weighted = f1_score(y_test, y_pred, average="weighted")
test_accuracy = accuracy_score(y_test, y_pred)

# AUC-ROC (one-vs-rest)
y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
test_auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")

print(f"\n  Test Results:")
print(f"  {'='*40}")
print(f"  Macro F1-score:    {test_f1_macro:.4f}")
print(f"  Weighted F1-score: {test_f1_weighted:.4f}")
print(f"  Accuracy:          {test_accuracy:.4f}")
print(f"  AUC-ROC (macro):   {test_auc:.4f}")

# Check against proposal target
target_f1 = 0.80
if test_f1_macro >= target_f1:
    print(f"\n  ✅ PASSES proposal target (macro-F1 ≥ {target_f1})")
else:
    print(f"\n  ⚠️  Below proposal target (macro-F1 ≥ {target_f1}) — DL models should improve this")

# Full classification report
print(f"\n  Per-Class Classification Report:")
print(f"  {'-'*60}")
report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
print(report)

# ============================================================
# FIGURE 1: Confusion Matrix
# ============================================================
print("Generating figures...")

cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0], linewidths=0.5)
axes[0].set_title("Confusion Matrix (Counts)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
axes[0].tick_params(axis="x", rotation=30)
axes[0].tick_params(axis="y", rotation=0)

# Normalized (percentages)
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
axes[1].set_title("Confusion Matrix (Normalized)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
axes[1].tick_params(axis="x", rotation=30)
axes[1].tick_params(axis="y", rotation=0)

plt.suptitle(f"Logistic Regression — Test Macro-F1: {test_f1_macro:.4f}",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")

# ============================================================
# FIGURE 2: ROC Curves (One-vs-Rest)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

colors = ["#e74c3c", "#8e44ad", "#3498db", "#95a5a6", "#f39c12", "#2ecc71"]

for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    cls_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
            label=f"{cls} (AUC={cls_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC=0.500)")
ax.set_title(f"ROC Curves — Logistic Regression (Macro AUC={test_auc:.4f})",
             fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "roc_curves.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")

# ============================================================
# FIGURE 3: Per-Class F1 Scores Bar Chart
# ============================================================
per_class_f1 = f1_score(y_test, y_pred, average=None)

fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = [colors[i % len(colors)] for i in range(n_classes)]
bars = ax.bar(class_names, per_class_f1, color=bar_colors, edgecolor="white")

ax.axhline(y=test_f1_macro, color="black", linestyle="--", alpha=0.5,
           label=f"Macro F1 = {test_f1_macro:.3f}")
ax.axhline(y=target_f1, color="red", linestyle="--", alpha=0.5,
           label=f"Target = {target_f1:.2f}")

for i, (cls, f1) in enumerate(zip(class_names, per_class_f1)):
    ax.text(i, f1 + 0.01, f"{f1:.3f}", ha="center", fontsize=10)

ax.set_title("Per-Class F1 Score — Logistic Regression", fontsize=13, fontweight="bold")
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_ylim(0, 1.1)
ax.tick_params(axis="x", rotation=30)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "per_class_f1.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")

# ============================================================
# FEATURE IMPORTANCE: Top Genes from Model Coefficients
# ============================================================
print("\n" + "=" * 60)
print("Feature Importance: Top genes from LR coefficients...")
print("=" * 60)

# For logistic regression, the coefficients directly tell us
# which genes are most important for each class
# Shape: (n_classes, n_features)
coefs = best_model.coef_

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, cls in enumerate(class_names):
    if i >= 6:
        break

    # Top 15 genes with highest absolute coefficient for this class
    cls_coefs = coefs[i]
    top_idx = np.argsort(np.abs(cls_coefs))[::-1][:15]
    top_genes = gene_names[top_idx]
    top_values = cls_coefs[top_idx]

    # Color positive (pushes toward this class) and negative (pushes away)
    bar_colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_values]

    axes[i].barh(range(15), top_values[::-1], color=bar_colors[::-1], edgecolor="white")
    axes[i].set_yticks(range(15))
    axes[i].set_yticklabels(top_genes[::-1], fontsize=9)
    axes[i].set_title(f"{cls}", fontsize=11, fontweight="bold")
    axes[i].set_xlabel("Coefficient", fontsize=9)
    axes[i].axvline(x=0, color="gray", linewidth=0.5)

    # Highlight known markers
    known_markers = {
        "Naive": ["CCR7", "SELL", "TCF7", "LEF1"],
        "Effector": ["GZMB", "GZMA", "PRF1", "NKG7"],
        "Exhausted": ["PDCD1", "HAVCR2", "LAG3", "TOX", "LAYN"],
        "Treg": ["FOXP3", "CTLA4", "IL2RA"],
        "Th1-like": ["CXCL13", "GZMK", "IFNG"],
        "Other_CD4": []
    }
    expected = known_markers.get(cls, [])
    found_in_top = [g for g in top_genes if g in expected]
    if found_in_top:
        axes[i].set_title(f"{cls} (✓ {', '.join(found_in_top)})", fontsize=10, fontweight="bold")

# Hide unused
for i in range(len(class_names), len(axes)):
    axes[i].set_visible(False)

plt.suptitle("Top 15 Genes per Class — Logistic Regression Coefficients\n(Red = positive, Blue = negative)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "feature_importance.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")

# Print top genes per class in text
print(f"\n  Top 10 genes per class (by |coefficient|):")
for i, cls in enumerate(class_names):
    cls_coefs = coefs[i]
    top_idx = np.argsort(np.abs(cls_coefs))[::-1][:10]
    top_genes_list = gene_names[top_idx]
    print(f"\n  {cls}:")
    for j, (gene, coef) in enumerate(zip(top_genes_list, cls_coefs[top_idx])):
        direction = "↑" if coef > 0 else "↓"
        print(f"    {j+1:2d}. {gene:<12} {direction} ({coef:+.4f})")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("Saving results...")
print("=" * 60)

results = {
    "model": "Logistic Regression",
    "best_params": search.best_params_,
    "cv_macro_f1_mean": float(mean_f1),
    "cv_macro_f1_std": float(std_f1),
    "cv_accuracy_mean": float(mean_acc),
    "cv_accuracy_std": float(std_acc),
    "test_macro_f1": float(test_f1_macro),
    "test_weighted_f1": float(test_f1_weighted),
    "test_accuracy": float(test_accuracy),
    "test_auc_roc": float(test_auc),
    "per_class_f1": {cls: float(f) for cls, f in zip(class_names, per_class_f1)},
    "fold_scores": fold_scores,
    "passes_target": bool(test_f1_macro >= target_f1)
}

with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Saved: results.json")

# Save model for later comparison
import pickle
with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
print(f"  Saved: model.pkl")

# Save predictions for later analysis
np.save(os.path.join(OUTPUT_DIR, "y_pred.npy"), y_pred)
np.save(os.path.join(OUTPUT_DIR, "y_proba.npy"), y_proba)
print(f"  Saved: y_pred.npy, y_proba.npy")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 5A COMPLETE — Logistic Regression Baseline")
print("=" * 60)
print(f"""
  ┌─────────────────────────────────────────────┐
  │  LOGISTIC REGRESSION RESULTS                │
  │                                             │
  │  CV Macro-F1:    {mean_f1:.4f} ± {std_f1:.4f}          │
  │  Test Macro-F1:  {test_f1_macro:.4f}                   │
  │  Test Accuracy:  {test_accuracy:.4f}                   │
  │  Test AUC-ROC:   {test_auc:.4f}                   │
  │                                             │
  │  Target (≥0.80): {"✅ PASS" if test_f1_macro >= 0.80 else "⚠️  BELOW — DL should improve"}            │
  └─────────────────────────────────────────────┘

  Per-class F1:
    {chr(10).join(f'    {cls:<12}: {f:.4f}' for cls, f in zip(class_names, per_class_f1))}

  Files in {OUTPUT_DIR}/:
    confusion_matrix.png    — Confusion matrix (counts + normalized)
    roc_curves.png          — ROC curves per class
    per_class_f1.png        — F1 bar chart per class
    feature_importance.png  — Top 15 genes per class
    results.json            — All metrics (for comparison table later)
    model.pkl               — Trained model
    y_pred.npy, y_proba.npy — Predictions

  This is your FIRST BASELINE. The number to beat is:
    Macro-F1 = {test_f1_macro:.4f}

  Next: Run the same for SVM, RF, XGBoost, then deep learning models.
  If LR already hits ≥ 0.80, that's great — it means the data is clean
  and the classes are separable. DL should push it even higher.
""")