"""
ImmunoNet — Step 6: Deep Learning Models (MLP, 1D-CNN, Self-Attention)
========================================================================
This script:
  1. Builds and trains MLP (4 layers, BatchNorm, Dropout)
  2. Builds and trains 1D-CNN (2 conv layers, global avg pooling)
  3. Builds and trains Self-Attention Classifier (multi-head attention)
  4. Evaluates all three on held-out test set
  5. Generates confusion matrices, ROC curves, training curves
  6. Compares DL models against best classical baseline
  7. Saves models, predictions, and attention weights

Requirements: pip install numpy tensorflow scikit-learn matplotlib seaborn

Input:  step3_*.npy files
Output: Results and figures in results/<model_name>/
        Comparison in results/dl_comparison/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, Model
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
RANDOM_STATE = 42
EPOCHS = 200          # Max epochs (early stopping will cut this short)
BATCH_SIZE = 64
PATIENCE = 15         # Stop if no improvement for 15 epochs
LEARNING_RATE = 1e-3

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("Loading preprocessed data...")
print("=" * 60)

X_train = np.load("step3_X_train.npy").astype(np.float32)
X_test = np.load("step3_X_test.npy").astype(np.float32)
y_train = np.load("step3_y_train.npy")
y_test = np.load("step3_y_test.npy")
gene_names = np.load("step3_gene_names.npy", allow_pickle=True)
class_weights_arr = np.load("step3_class_weights.npy")

with open("step3_label_mapping.json") as f:
    label_map = json.load(f)

class_names = [label_map[str(i)] for i in range(len(label_map))]
n_classes = len(class_names)
n_features = X_train.shape[1]

# Class weight dict for Keras
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights_arr)}

# Validation split from training data (85% train, 15% val)
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15,
    random_state=RANDOM_STATE, stratify=y_train
)

print(f"  Training:   {X_tr.shape}")
print(f"  Validation: {X_val.shape}")
print(f"  Test:       {X_test.shape}")
print(f"  Features:   {n_features}")
print(f"  Classes:    {n_classes} → {class_names}")
print(f"  GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")


# ============================================================
# HELPER: Evaluate and save results for a DL model
# ============================================================
def evaluate_dl_model(model, model_name, history, X_test, y_test,
                      class_names, n_classes, output_dir):
    """Evaluate DL model and generate all figures."""

    os.makedirs(output_dir, exist_ok=True)

    # Predictions
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    # Metrics
    test_f1_macro = f1_score(y_test, y_pred, average="macro")
    test_f1_weighted = f1_score(y_test, y_pred, average="weighted")
    test_accuracy = accuracy_score(y_test, y_pred)
    per_class_f1 = f1_score(y_test, y_pred, average=None)

    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    test_auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")

    print(f"\n  {model_name} — Test Results:")
    print(f"  {'='*40}")
    print(f"  Macro F1:    {test_f1_macro:.4f}")
    print(f"  Accuracy:    {test_accuracy:.4f}")
    print(f"  AUC-ROC:     {test_auc:.4f}")
    print(f"\n  Per-Class F1:")
    for cls, f in zip(class_names, per_class_f1):
        print(f"    {cls:<12}: {f:.4f}")

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(f"\n{report}")

    # --- Figure 1: Training Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Loss Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train Acc", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Acc", linewidth=2)
    axes[1].set_title("Accuracy Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    best_epoch = np.argmin(history.history["val_loss"]) + 1
    axes[0].axvline(x=best_epoch-1, color="red", linestyle="--", alpha=0.5, label=f"Best epoch: {best_epoch}")
    axes[0].legend()

    plt.suptitle(f"{model_name} — Training History (stopped at epoch {len(history.history['loss'])})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure 2: Confusion Matrix ---
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

    plt.suptitle(f"{model_name} — Macro-F1: {test_f1_macro:.4f}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure 3: ROC Curves ---
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c", "#8e44ad", "#3498db", "#95a5a6", "#f39c12", "#2ecc71"]

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
        "test_macro_f1": float(test_f1_macro),
        "test_weighted_f1": float(test_f1_weighted),
        "test_accuracy": float(test_accuracy),
        "test_auc_roc": float(test_auc),
        "per_class_f1": {cls: float(f) for cls, f in zip(class_names, per_class_f1)},
        "epochs_trained": len(history.history["loss"]),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(min(history.history["val_loss"])),
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    model.save(os.path.join(output_dir, "model.keras"))
    np.save(os.path.join(output_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(output_dir, "y_proba.npy"), y_proba)

    print(f"\n  Results saved to {output_dir}/")
    return results


# ============================================================
# COMMON CALLBACKS
# ============================================================
def get_callbacks(model_name):
    """Standard callbacks for all DL models."""
    log_dir = f"results/{model_name}/logs"
    os.makedirs(log_dir, exist_ok=True)

    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.TensorBoard(log_dir=log_dir),
    ]


# ============================================================
# MODEL 1: Multi-Layer Perceptron (MLP)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 1: Multi-Layer Perceptron (MLP)")
print("  Architecture: Input(3000) → 512 → 256 → 128 → 6")
print("  BatchNorm + ReLU + Dropout(0.3) between each layer")
print("=" * 60)

start = time.time()

mlp_model = keras.Sequential([
    # Input layer
    layers.Input(shape=(n_features,)),

    # Hidden layer 1: 512 units
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.3),

    # Hidden layer 2: 256 units
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.3),

    # Hidden layer 3: 128 units
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.3),

    # Output layer: 6 classes
    layers.Dense(n_classes, activation="softmax")
], name="MLP")

mlp_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

mlp_model.summary()

print(f"\n  Training MLP...")
mlp_history = mlp_model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=get_callbacks("mlp"),
    verbose=1
)

print(f"  MLP training time: {time.time()-start:.1f}s")

mlp_results = evaluate_dl_model(
    model=mlp_model,
    model_name="MLP",
    history=mlp_history,
    X_test=X_test, y_test=y_test,
    class_names=class_names, n_classes=n_classes,
    output_dir="results/mlp"
)


# ============================================================
# MODEL 2: 1D Convolutional Neural Network (1D-CNN)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 2: 1D-CNN")
print("  Architecture: Conv1D(32) → Conv1D(64) → GlobalAvgPool → Dense")
print("  Treats gene expression vector as a 1D signal")
print("=" * 60)

start = time.time()

# Reshape for Conv1D: (batch, features, 1) — treat each gene as a timestep with 1 channel
X_tr_cnn = X_tr.reshape(-1, n_features, 1)
X_val_cnn = X_val.reshape(-1, n_features, 1)
X_test_cnn = X_test.reshape(-1, n_features, 1)

cnn_model = keras.Sequential([
    layers.Input(shape=(n_features, 1)),

    # Conv block 1: 32 filters, kernel size 7
    layers.Conv1D(32, kernel_size=7, padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),

    # Conv block 2: 64 filters, kernel size 5
    layers.Conv1D(64, kernel_size=5, padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),

    # Conv block 3: 128 filters, kernel size 3
    layers.Conv1D(128, kernel_size=3, padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    # Global Average Pooling — compress to fixed-size vector
    layers.GlobalAveragePooling1D(),

    # Dense classification head
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.3),

    layers.Dense(n_classes, activation="softmax")
], name="CNN_1D")

cnn_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cnn_model.summary()

print(f"\n  Training 1D-CNN...")
cnn_history = cnn_model.fit(
    X_tr_cnn, y_tr,
    validation_data=(X_val_cnn, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=get_callbacks("cnn_1d"),
    verbose=1
)

print(f"  1D-CNN training time: {time.time()-start:.1f}s")

# Evaluate CNN (needs reshaped test data)
# Temporarily override predict to handle reshape
cnn_y_proba = cnn_model.predict(X_test_cnn, verbose=0)
cnn_y_pred = np.argmax(cnn_y_proba, axis=1)

cnn_results = evaluate_dl_model(
    model=cnn_model,
    model_name="1D-CNN",
    history=cnn_history,
    X_test=X_test_cnn, y_test=y_test,
    class_names=class_names, n_classes=n_classes,
    output_dir="results/cnn_1d"
)


# ============================================================
# MODEL 3: Self-Attention Classifier
# ============================================================
print("\n" + "=" * 60)
print("MODEL 3: Self-Attention Classifier")
print("  Architecture: GeneEmbedding → MultiHeadAttention → Dense")
print("  Learns which genes are most important for each prediction")
print("=" * 60)

start = time.time()


class GeneAttentionClassifier(Model):
    """
    Self-Attention model for gene expression classification.

    How it works:
    1. Each gene's expression value is projected into a higher-dimensional embedding
    2. Multi-head self-attention lets every gene "look at" every other gene
       to learn which combinations matter for classification
    3. Attention weights tell us which genes the model focuses on (interpretability)
    4. Global average pooling aggregates all gene embeddings into one vector
    5. Dense layers make the final classification
    """

    def __init__(self, n_features, n_classes, embed_dim=64, num_heads=4, ff_dim=128, dropout_rate=0.3):
        super().__init__()

        self.n_features = n_features
        self.embed_dim = embed_dim

        # Project each gene's scalar expression into embed_dim dimensions
        self.gene_projection = layers.Dense(embed_dim)
        self.projection_norm = layers.LayerNormalization()

        # Positional encoding (learnable)
        self.pos_embedding = layers.Embedding(input_dim=n_features, output_dim=embed_dim)

        # Multi-head self-attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
        )
        self.attention_norm = layers.LayerNormalization()
        self.attention_dropout = layers.Dropout(dropout_rate)

        # Feed-forward network after attention
        self.ff_dense1 = layers.Dense(ff_dim, activation="relu")
        self.ff_dense2 = layers.Dense(embed_dim)
        self.ff_norm = layers.LayerNormalization()
        self.ff_dropout = layers.Dropout(dropout_rate)

        # Global pooling + classification head
        self.global_pool = layers.GlobalAveragePooling1D()
        self.classifier_dense = layers.Dense(128, activation="relu")
        self.classifier_dropout = layers.Dropout(dropout_rate)
        self.classifier_output = layers.Dense(n_classes, activation="softmax")

    def call(self, inputs, training=False, return_attention=False):
        # inputs shape: (batch, n_features) — flat gene expression vector

        # Reshape to (batch, n_features, 1) then project to (batch, n_features, embed_dim)
        x = tf.expand_dims(inputs, axis=-1)
        x = self.gene_projection(x)
        x = self.projection_norm(x)

        # Add positional encoding
        positions = tf.range(start=0, limit=self.n_features, delta=1)
        pos_embed = self.pos_embedding(positions)
        x = x + pos_embed

        # Multi-head self-attention with residual connection
        attn_output, attn_weights = self.attention(
            query=x, key=x, value=x,
            return_attention_scores=True,
            training=training
        )
        attn_output = self.attention_dropout(attn_output, training=training)
        x = self.attention_norm(x + attn_output)

        # Feed-forward with residual connection
        ff_output = self.ff_dense1(x)
        ff_output = self.ff_dense2(ff_output)
        ff_output = self.ff_dropout(ff_output, training=training)
        x = self.ff_norm(x + ff_output)

        # Global average pooling → classification
        x = self.global_pool(x)
        x = self.classifier_dense(x)
        x = self.classifier_dropout(x, training=training)
        output = self.classifier_output(x)

        if return_attention:
            return output, attn_weights
        return output


# Build and compile
attn_model = GeneAttentionClassifier(
    n_features=n_features,
    n_classes=n_classes,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    dropout_rate=0.3
)

# Build the model by passing dummy input
_ = attn_model(tf.zeros((1, n_features)))
attn_model.summary()

attn_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"\n  Training Self-Attention Classifier...")
attn_history = attn_model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=get_callbacks("attention"),
    verbose=1
)

print(f"  Attention training time: {time.time()-start:.1f}s")

attn_results = evaluate_dl_model(
    model=attn_model,
    model_name="Self-Attention",
    history=attn_history,
    X_test=X_test, y_test=y_test,
    class_names=class_names, n_classes=n_classes,
    output_dir="results/attention"
)

# ============================================================
# EXTRACT ATTENTION WEIGHTS (Interpretability)
# ============================================================
print("\n" + "=" * 60)
print("Extracting attention weights for interpretability...")
print("=" * 60)

# Get attention weights for test set (process in batches to save memory)
batch_size = 256
all_attn_weights = []

for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    _, attn_w = attn_model(batch, return_attention=True)
    # attn_w shape: (batch, num_heads, n_features, n_features)
    # Average across heads and queries to get per-gene importance
    gene_importance = tf.reduce_mean(attn_w, axis=[1, 2]).numpy()  # (batch, n_features)
    all_attn_weights.append(gene_importance)

attn_gene_importance = np.vstack(all_attn_weights)  # (n_test_cells, n_features)

# Average attention across all test cells per class
attn_per_class = {}
for i, cls in enumerate(class_names):
    mask = y_test == i
    if mask.sum() > 0:
        cls_attn = attn_gene_importance[mask].mean(axis=0)
        top_idx = np.argsort(cls_attn)[::-1][:15]
        attn_per_class[cls] = {
            "top_genes": gene_names[top_idx].tolist(),
            "top_scores": cls_attn[top_idx].tolist()
        }
        print(f"  {cls} — Top 5 attention genes: {gene_names[top_idx][:5].tolist()}")

# Save attention weights
np.save("results/attention/attention_gene_importance.npy", attn_gene_importance)
with open("results/attention/attention_per_class.json", "w") as f:
    json.dump(attn_per_class, f, indent=2)

# --- Attention Heatmap ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, cls in enumerate(class_names):
    if idx >= 6:
        break
    data = attn_per_class[cls]
    genes = data["top_genes"][:15]
    scores = data["top_scores"][:15]

    axes[idx].barh(range(len(genes)), scores[::-1], color="#8e44ad", edgecolor="white")
    axes[idx].set_yticks(range(len(genes)))
    axes[idx].set_yticklabels(genes[::-1], fontsize=9)
    axes[idx].set_title(f"{cls}", fontsize=11, fontweight="bold")
    axes[idx].set_xlabel("Attention Score", fontsize=9)

for idx in range(len(class_names), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Top 15 Genes by Attention Weight per Class — Self-Attention Model",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/attention/attention_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: results/attention/attention_heatmap.png")


# ============================================================
# COMPARISON: All DL Models + Best Classical
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON: Deep Learning Models + Best Classical Baseline")
print("=" * 60)

comp_dir = "results/dl_comparison"
os.makedirs(comp_dir, exist_ok=True)

# Load best classical result
all_results = []
classical_path = "results/classical_comparison/all_classical_results.json"
if os.path.exists(classical_path):
    with open(classical_path) as f:
        classical_results = json.load(f)
    # Find best classical
    best_classical = max(classical_results, key=lambda x: x["test_macro_f1"])
    all_results.append(best_classical)
    print(f"  Best classical: {best_classical['model']} (F1={best_classical['test_macro_f1']:.4f})")
else:
    # Load LR as fallback
    lr_path = "results/logistic_regression/results.json"
    if os.path.exists(lr_path):
        with open(lr_path) as f:
            best_classical = json.load(f)
        all_results.append(best_classical)

# Add DL results
all_results.append(mlp_results)
all_results.append(cnn_results)
all_results.append(attn_results)

# Print comparison table
print(f"\n  {'Model':<22} {'Test F1':>10} {'Test Acc':>10} {'AUC-ROC':>10}")
print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
for r in all_results:
    auc_str = f"{r['test_auc_roc']:.4f}" if r.get("test_auc_roc") else "N/A"
    print(f"  {r['model']:<22} {r['test_macro_f1']:>10.4f} {r['test_accuracy']:>10.4f} {auc_str:>10}")

best_overall = max(all_results, key=lambda x: x["test_macro_f1"])
print(f"\n  🏆 Best Overall: {best_overall['model']} (F1={best_overall['test_macro_f1']:.4f})")

# --- Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(12, 6))

model_names = [r["model"] for r in all_results]
test_f1s = [r["test_macro_f1"] for r in all_results]
bar_colors = ["#95a5a6"] + ["#e74c3c", "#3498db", "#8e44ad"]  # gray for classical, colors for DL

bars = ax.bar(model_names, test_f1s, color=bar_colors[:len(model_names)], edgecolor="white", linewidth=0.5)
ax.axhline(y=0.80, color="red", linestyle="--", alpha=0.5, label="Target (0.80)")

for i, (name, f1) in enumerate(zip(model_names, test_f1s)):
    ax.text(i, f1 + 0.003, f"{f1:.4f}", ha="center", fontsize=11, fontweight="bold")

ax.set_title("Deep Learning vs Best Classical Baseline", fontsize=14, fontweight="bold")
ax.set_ylabel("Test Macro-F1", fontsize=12)
ax.set_ylim(0.75, 1.0)
ax.tick_params(axis="x", rotation=15)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(comp_dir, "dl_vs_classical.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {comp_dir}/dl_vs_classical.png")

# --- Per-Class Comparison ---
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(n_classes)
width = 0.18
offsets = np.arange(len(all_results)) - (len(all_results) - 1) / 2

for idx, r in enumerate(all_results):
    per_f1 = [r["per_class_f1"][cls] for cls in class_names]
    ax.bar(x + offsets[idx] * width, per_f1, width,
           label=r["model"], color=bar_colors[idx],
           edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=30, fontsize=10)
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_title("Per-Class F1 — DL vs Classical", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0.65, 1.05)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(comp_dir, "per_class_dl_vs_classical.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {comp_dir}/per_class_dl_vs_classical.png")

# Save all results
with open(os.path.join(comp_dir, "all_dl_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 6 COMPLETE — All Deep Learning Models Trained")
print("=" * 60)
print(f"""
  ┌────────────────────────────────────────────────────────────┐
  │  DEEP LEARNING RESULTS                                    │
  │                                                            │""")
for r in all_results:
    marker = "🏆" if r == best_overall else "  "
    print(f"  │  {marker} {r['model']:<22} Macro-F1: {r['test_macro_f1']:.4f}             │")
print(f"""  │                                                            │
  │  DL improvement over classical: {all_results[-1]['test_macro_f1'] - all_results[0]['test_macro_f1']:+.4f}                  │
  └────────────────────────────────────────────────────────────┘

  Files created:
    results/mlp/
      ├── training_curves.png, confusion_matrix.png, roc_curves.png
      ├── results.json, model.keras, y_pred.npy
    results/cnn_1d/
      ├── training_curves.png, confusion_matrix.png, roc_curves.png
      ├── results.json, model.keras, y_pred.npy
    results/attention/
      ├── training_curves.png, confusion_matrix.png, roc_curves.png
      ├── attention_heatmap.png  ← which genes the model focuses on
      ├── attention_gene_importance.npy, attention_per_class.json
      ├── results.json, model.keras, y_pred.npy
    results/dl_comparison/
      ├── dl_vs_classical.png, per_class_dl_vs_classical.png
      └── all_dl_results.json

  Next steps:
    - Step 7: SHAP interpretability analysis
    - Step 8: Cross-dataset validation (GSE126030)
    - Step 9: Final comparison table + statistical testing
""")