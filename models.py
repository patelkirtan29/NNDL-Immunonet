import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import json
import os
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
N_CV_FOLDS = 5

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
class_weights_arr = np.load("step3_class_weights.npy")

with open("step3_label_mapping.json") as f:
    label_map = json.load(f)

class_names = [label_map[str(i)] for i in range(len(label_map))]
n_classes = len(class_names)

# Class weight dict for XGBoost
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}

cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  Classes: {class_names}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test,
                   class_names, n_classes, label_map, cv, output_dir):
    """Train, evaluate, and save results for a single model."""

    os.makedirs(output_dir, exist_ok=True)

    # --- Cross-validation ---
    print(f"\n  5-Fold Cross-Validation:")
    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_f_train, X_f_val = X_train[train_idx], X_train[val_idx]
        y_f_train, y_f_val = y_train[train_idx], y_train[val_idx]

        # Clone model with same params
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_f_train, y_f_train)
        y_f_pred = fold_model.predict(X_f_val)

        fold_f1 = f1_score(y_f_val, y_f_pred, average="macro")
        fold_acc = accuracy_score(y_f_val, y_f_pred)
        fold_scores.append({"fold": fold_idx + 1, "macro_f1": fold_f1, "accuracy": fold_acc})
        print(f"    Fold {fold_idx+1}: macro-F1={fold_f1:.4f}, accuracy={fold_acc:.4f}")

    mean_f1 = np.mean([s["macro_f1"] for s in fold_scores])
    std_f1 = np.std([s["macro_f1"] for s in fold_scores])
    mean_acc = np.mean([s["accuracy"] for s in fold_scores])
    std_acc = np.std([s["accuracy"] for s in fold_scores])
    print(f"    CV macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")

    # --- Test set evaluation ---
    print(f"\n  Test Set Evaluation:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Get probabilities (different for SVM vs tree models)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        # Convert decision function to pseudo-probabilities via softmax
        from scipy.special import softmax
        y_proba = softmax(decision, axis=1)
    else:
        y_proba = None

    test_f1_macro = f1_score(y_test, y_pred, average="macro")
    test_f1_weighted = f1_score(y_test, y_pred, average="weighted")
    test_accuracy = accuracy_score(y_test, y_pred)
    per_class_f1 = f1_score(y_test, y_pred, average=None)

    # AUC-ROC
    test_auc = None
    if y_proba is not None:
        y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
        test_auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")

    print(f"    Macro F1:    {test_f1_macro:.4f}")
    print(f"    Accuracy:    {test_accuracy:.4f}")
    if test_auc:
        print(f"    AUC-ROC:     {test_auc:.4f}")

    print(f"\n  Per-Class F1:")
    for cls, f in zip(class_names, per_class_f1):
        print(f"    {cls:<12}: {f:.4f}")

    # --- Classification Report ---
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(f"\n{report}")

    # --- Figure 1: Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Counts", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis="x", rotation=30)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title("Normalized", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle(f"{model_name} — Test Macro-F1: {test_f1_macro:.4f}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure 2: ROC Curves ---
    if y_proba is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#e74c3c", "#8e44ad", "#3498db", "#95a5a6", "#f39c12", "#2ecc71"]
        y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            cls_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                    label=f"{cls} (AUC={cls_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title(f"ROC Curves — {model_name} (Macro AUC={test_auc:.4f})",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # --- Save results ---
    results = {
        "model": model_name,
        "cv_macro_f1_mean": float(mean_f1),
        "cv_macro_f1_std": float(std_f1),
        "cv_accuracy_mean": float(mean_acc),
        "cv_accuracy_std": float(std_acc),
        "test_macro_f1": float(test_f1_macro),
        "test_weighted_f1": float(test_f1_weighted),
        "test_accuracy": float(test_accuracy),
        "test_auc_roc": float(test_auc) if test_auc else None,
        "per_class_f1": {cls: float(f) for cls, f in zip(class_names, per_class_f1)},
        "fold_scores": fold_scores,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    np.save(os.path.join(output_dir, "y_pred.npy"), y_pred)
    if y_proba is not None:
        np.save(os.path.join(output_dir, "y_proba.npy"), y_proba)

    print(f"  Results saved to {output_dir}/")

    return results


# ============================================================
# MODEL 1: SVM (Linear)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 1: SVM (Linear Kernel)")
print("=" * 60)

start = time.time()

svm_linear_params = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "kernel": ["linear"],
    "class_weight": ["balanced"],
    "max_iter": [2000],
}

svm_linear_base = SVC(random_state=RANDOM_STATE, decision_function_shape="ovr")

search_svm_linear = RandomizedSearchCV(
    estimator=svm_linear_base,
    param_distributions=svm_linear_params,
    n_iter=6,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)
search_svm_linear.fit(X_train, y_train)

print(f"\n  Best CV macro-F1: {search_svm_linear.best_score_:.4f}")
print(f"  Best params: {search_svm_linear.best_params_}")
print(f"  Time: {time.time()-start:.1f}s")

svm_linear_results = evaluate_model(
    model=search_svm_linear.best_estimator_,
    model_name="SVM (Linear)",
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    class_names=class_names, n_classes=n_classes,
    label_map=label_map, cv=cv,
    output_dir="results/svm_linear"
)

# ============================================================
# MODEL 2: SVM (RBF Kernel)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 2: SVM (RBF Kernel)")
print("=" * 60)

start = time.time()

svm_rbf_params = {
    "C": [0.01, 0.1, 1, 10, 100],
    "kernel": ["rbf"],
    "gamma": ["scale", "auto"],
    "class_weight": ["balanced"],
    "max_iter": [2000],
}

svm_rbf_base = SVC(random_state=RANDOM_STATE, decision_function_shape="ovr")

search_svm_rbf = RandomizedSearchCV(
    estimator=svm_rbf_base,
    param_distributions=svm_rbf_params,
    n_iter=10,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)
search_svm_rbf.fit(X_train, y_train)

print(f"\n  Best CV macro-F1: {search_svm_rbf.best_score_:.4f}")
print(f"  Best params: {search_svm_rbf.best_params_}")
print(f"  Time: {time.time()-start:.1f}s")

svm_rbf_results = evaluate_model(
    model=search_svm_rbf.best_estimator_,
    model_name="SVM (RBF)",
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    class_names=class_names, n_classes=n_classes,
    label_map=label_map, cv=cv,
    output_dir="results/svm_rbf"
)

# ============================================================
# MODEL 3: Random Forest
# ============================================================
print("\n" + "=" * 60)
print("MODEL 3: Random Forest")
print("=" * 60)

start = time.time()

rf_params = {
    "n_estimators": [100, 200, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced"],
}

rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

search_rf = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=rf_params,
    n_iter=20,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)
search_rf.fit(X_train, y_train)

print(f"\n  Best CV macro-F1: {search_rf.best_score_:.4f}")
print(f"  Best params: {search_rf.best_params_}")
print(f"  Time: {time.time()-start:.1f}s")

rf_results = evaluate_model(
    model=search_rf.best_estimator_,
    model_name="Random Forest",
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    class_names=class_names, n_classes=n_classes,
    label_map=label_map, cv=cv,
    output_dir="results/random_forest"
)

# ============================================================
# MODEL 4: XGBoost
# ============================================================
print("\n" + "=" * 60)
print("MODEL 4: XGBoost")
print("=" * 60)

try:
    from xgboost import XGBClassifier

    start = time.time()

    # Compute sample weights for XGBoost (doesn't use class_weight param)
    sample_weights_train = np.array([class_weight_dict[y] for y in y_train])

    xgb_params = {
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [1, 2, 5],
    }

    xgb_base = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )

    search_xgb = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_params,
        n_iter=20,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    search_xgb.fit(X_train, y_train, sample_weight=sample_weights_train)

    print(f"\n  Best CV macro-F1: {search_xgb.best_score_:.4f}")
    print(f"  Best params: {search_xgb.best_params_}")
    print(f"  Time: {time.time()-start:.1f}s")

    xgb_results = evaluate_model(
        model=search_xgb.best_estimator_,
        model_name="XGBoost",
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        class_names=class_names, n_classes=n_classes,
        label_map=label_map, cv=cv,
        output_dir="results/xgboost"
    )

except ImportError:
    print("  XGBoost not installed. Run: pip install xgboost")
    print("  Skipping XGBoost...")
    xgb_results = None

# ============================================================
# COMPARISON TABLE: All Classical Models
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON TABLE: All Classical Models")
print("=" * 60)

# Load LR results from Step 5a
all_results = []
lr_results_path = "results/logistic_regression/results.json"
if os.path.exists(lr_results_path):
    with open(lr_results_path) as f:
        lr_results = json.load(f)
    all_results.append(lr_results)

all_results.append(svm_linear_results)
all_results.append(svm_rbf_results)
all_results.append(rf_results)
if xgb_results:
    all_results.append(xgb_results)

# Build comparison table
comp_dir = "results/classical_comparison"
os.makedirs(comp_dir, exist_ok=True)

print(f"\n  {'Model':<22} {'CV F1':>12} {'Test F1':>10} {'Test Acc':>10} {'AUC-ROC':>10}")
print(f"  {'-'*22} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

table_data = []
for r in all_results:
    cv_str = f"{r['cv_macro_f1_mean']:.4f}±{r['cv_macro_f1_std']:.4f}"
    auc_str = f"{r['test_auc_roc']:.4f}" if r.get('test_auc_roc') else "N/A"
    print(f"  {r['model']:<22} {cv_str:>12} {r['test_macro_f1']:>10.4f} {r['test_accuracy']:>10.4f} {auc_str:>10}")

    table_data.append({
        "Model": r["model"],
        "CV Macro-F1": cv_str,
        "Test Macro-F1": r["test_macro_f1"],
        "Test Accuracy": r["test_accuracy"],
        "AUC-ROC": r.get("test_auc_roc", None)
    })

# Find best model
best_result = max(all_results, key=lambda x: x["test_macro_f1"])
print(f"\n  🏆 Best Classical Model: {best_result['model']} (Test Macro-F1: {best_result['test_macro_f1']:.4f})")

# ============================================================
# FIGURE: Comparison Bar Chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

model_names = [r["model"] for r in all_results]
test_f1s = [r["test_macro_f1"] for r in all_results]
test_accs = [r["test_accuracy"] for r in all_results]
bar_colors = ["#3498db", "#e74c3c", "#8e44ad", "#2ecc71", "#f39c12"]

# Macro-F1 comparison
bars1 = axes[0].bar(model_names, test_f1s, color=bar_colors[:len(model_names)], edgecolor="white")
axes[0].axhline(y=0.80, color="red", linestyle="--", alpha=0.5, label="Target (0.80)")
axes[0].set_title("Test Macro-F1 Score", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Macro-F1")
axes[0].set_ylim(0.70, 1.0)
axes[0].tick_params(axis="x", rotation=20)
axes[0].legend()

for i, (name, f1) in enumerate(zip(model_names, test_f1s)):
    axes[0].text(i, f1 + 0.005, f"{f1:.4f}", ha="center", fontsize=9, fontweight="bold")

# Accuracy comparison
bars2 = axes[1].bar(model_names, test_accs, color=bar_colors[:len(model_names)], edgecolor="white")
axes[1].set_title("Test Accuracy", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0.70, 1.0)
axes[1].tick_params(axis="x", rotation=20)

for i, (name, acc) in enumerate(zip(model_names, test_accs)):
    axes[1].text(i, acc + 0.005, f"{acc:.4f}", ha="center", fontsize=9, fontweight="bold")

plt.suptitle("Classical ML Baselines — Model Comparison",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
path = os.path.join(comp_dir, "model_comparison.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n  Saved comparison chart: {path}")

# ============================================================
# FIGURE: Per-Class F1 Comparison Across Models
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(n_classes)
width = 0.15
offsets = np.arange(len(all_results)) - (len(all_results) - 1) / 2

for idx, r in enumerate(all_results):
    per_f1 = [r["per_class_f1"][cls] for cls in class_names]
    ax.bar(x + offsets[idx] * width, per_f1, width,
           label=r["model"], color=bar_colors[idx % len(bar_colors)],
           edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=30, fontsize=10)
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_title("Per-Class F1 Scores — All Classical Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0.65, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=0.80, color="red", linestyle="--", alpha=0.3)

plt.tight_layout()
path = os.path.join(comp_dir, "per_class_comparison.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved per-class chart: {path}")

# Save comparison table as CSV
df_comparison = pd.DataFrame(table_data)
csv_path = os.path.join(comp_dir, "comparison_table.csv")
df_comparison.to_csv(csv_path, index=False)
print(f"  Saved comparison CSV: {csv_path}")

# Save all results for DL comparison later
with open(os.path.join(comp_dir, "all_classical_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 5B COMPLETE — All Classical Baselines Trained")
print("=" * 60)
print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  CLASSICAL ML RESULTS SUMMARY                           │
  │                                                          │""")

for r in all_results:
    name = r["model"]
    f1 = r["test_macro_f1"]
    marker = "🏆" if r == best_result else "  "
    print(f"  │  {marker} {name:<20} Macro-F1: {f1:.4f}               │")

print(f"""  │                                                          │
  │  Target (≥0.80): ALL MODELS PASS ✅                     │
  │  Best model: {best_result['model']:<20}                       │
  │  Number to beat with DL: {best_result['test_macro_f1']:.4f}                       │
  └──────────────────────────────────────────────────────────┘

  Files created:
    results/svm_linear/       — SVM Linear results + figures
    results/svm_rbf/          — SVM RBF results + figures
    results/random_forest/    — Random Forest results + figures
    results/xgboost/          — XGBoost results + figures
    results/classical_comparison/
      ├── model_comparison.png     — Side-by-side bar chart
      ├── per_class_comparison.png — Per-class F1 grouped bars
      ├── comparison_table.csv     — CSV for your report
      └── all_classical_results.json — All metrics for DL comparison

  Next step: Deep Learning models (MLP, 1D-CNN, Self-Attention)
  The bar to beat: {best_result['model']} at {best_result['test_macro_f1']:.4f}
""")