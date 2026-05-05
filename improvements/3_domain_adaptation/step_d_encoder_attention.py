"""
Step D: Encoder + Self-Attention Classifier
===========================================
Standalone model that learns an encoder-based latent representation with
self-attention over gene patches, then classifies source and target data.

This script does NOT retrain the Step A/B/C suite.
It is a separate architecture experiment for cross-dataset transfer.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step D: encoder + self-attention classifier")
    parser.add_argument("--confidence-threshold", type=float, default=0.55, help="Seed label confidence threshold")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Adam learning rate")
    parser.add_argument("--patch-size", type=int, default=25, help="Genes per patch")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Patch embedding size")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--encoder-dim", type=int, default=128, help="Encoder hidden width")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--max-iter", type=int, default=20, help="Max epochs for early stopping patience proxy")
    return parser.parse_args()


def pick_target_file(script_dir: Path) -> Path:
    correction_candidates = [
        script_dir.parent / "2_batch_correction/batch_correction_output/gse126030_corrected_coral.npy",
        script_dir.parent / "2_batch_correction/batch_correction_output/gse126030_corrected_combat.npy",
        script_dir.parent / "2_batch_correction/batch_correction_output/gse126030_corrected_mean_centered.npy",
        script_dir.parent / "2_batch_correction/batch_correction_output/gse126030_corrected_standardized.npy",
    ]
    for path in correction_candidates:
        if path.exists():
            return path
    fallback = script_dir.parent.parent / "gse126030_preprocessed.npy"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No GSE126030 target file found")


def load_data(project_root: Path, script_dir: Path, confidence_threshold: float):
    X_train = np.load(project_root / "step3_X_train.npy").astype(np.float32)
    X_test = np.load(project_root / "step3_X_test.npy").astype(np.float32)
    y_train = np.load(project_root / "step3_y_train.npy")
    y_test = np.load(project_root / "step3_y_test.npy")

    with open(project_root / "step3_label_mapping.json") as file_obj:
        label_map = json.load(file_obj)
    class_names = [label_map[str(index)] for index in range(len(label_map))]

    target_file = pick_target_file(script_dir)
    X_target = np.load(target_file).astype(np.float32)

    labels_path = script_dir.parent / "1_label_validation/label_validation_output/gse126030_reclustered_labels.csv"
    if labels_path.exists():
        df_labels = pd.read_csv(labels_path)
        y_target_names = df_labels["new_class"].values
        confidence = df_labels["confidence"].values
        target_mask = (confidence >= confidence_threshold) & (y_target_names != "Uncertain")
        X_target_filtered = X_target[target_mask]
        y_target_filtered = np.array([class_names.index(name) for name in y_target_names[target_mask]], dtype=np.int64)
    else:
        X_target_filtered = None
        y_target_filtered = None
        y_target_names = None

    return X_train, X_test, y_train, y_test, X_target, X_target_filtered, y_target_filtered, y_target_names, class_names, target_file


def build_model(n_genes: int, n_classes: int, args: argparse.Namespace):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(n_genes,), name="genes")

    x = layers.LayerNormalization()(inputs)
    x = layers.Dense(args.encoder_dim, activation="relu")(x)
    x = layers.Dropout(args.dropout)(x)
    x = layers.Dense(args.encoder_dim // 2, activation="relu")(x)
    x = layers.Dropout(args.dropout)(x)

    n_tokens = n_genes // args.patch_size
    truncated = n_tokens * args.patch_size
    token_x = layers.Lambda(lambda t: t[:, :truncated])(inputs)
    token_x = layers.Reshape((n_tokens, args.patch_size))(token_x)
    token_x = layers.Dense(args.embedding_dim)(token_x)

    positions = tf.range(start=0, limit=n_tokens, delta=1)
    pos_embedding = layers.Embedding(input_dim=n_tokens, output_dim=args.embedding_dim)
    token_x = token_x + pos_embedding(positions)
    token_x = layers.LayerNormalization()(token_x)

    for _ in range(2):
        attn = layers.MultiHeadAttention(num_heads=args.num_heads, key_dim=args.embedding_dim // args.num_heads)(token_x, token_x)
        attn = layers.Dropout(args.dropout)(attn)
        token_x = layers.Add()([token_x, attn])
        token_x = layers.LayerNormalization()(token_x)

        ff = layers.Dense(args.encoder_dim, activation="relu")(token_x)
        ff = layers.Dense(args.embedding_dim)(ff)
        ff = layers.Dropout(args.dropout)(ff)
        token_x = layers.Add()([token_x, ff])
        token_x = layers.LayerNormalization()(token_x)

    token_x = layers.GlobalAveragePooling1D()(token_x)
    token_x = layers.Dense(args.encoder_dim, activation="relu")(token_x)
    token_x = layers.Dropout(args.dropout)(token_x)
    outputs = layers.Dense(n_classes, activation="softmax", name="class_output")(token_x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="EncoderAttention")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_dir = script_dir / "improved_models_output"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("STEP D: ENCODER + SELF-ATTENTION CLASSIFIER")
    print("=" * 60)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_target,
        X_target_filtered,
        y_target_filtered,
        y_target_names,
        class_names,
        target_file,
    ) = load_data(project_root, script_dir, args.confidence_threshold)

    print(f"Loaded training: {X_train.shape}, test: {X_test.shape}")
    print(f"Target file: {target_file.name}")
    if X_target_filtered is not None:
        print(f"Filtered target labels: {X_target_filtered.shape}")

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import callbacks
    except ImportError as error:
        print("TensorFlow not available for Step D")
        raise error

    model = build_model(X_train.shape[1], len(class_names), args)
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight={i: w for i, w in enumerate(np.load(project_root / "step3_class_weights.npy"))},
        callbacks=cb,
        verbose=0,
    )

    y_pred_test = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_f1 = f1_score(y_test, y_pred_test, average="macro")
    test_acc = accuracy_score(y_test, y_pred_test)

    cross_f1 = None
    cross_acc = None
    cross_report = None
    if X_target_filtered is not None and y_target_filtered is not None:
        y_pred_target = np.argmax(model.predict(X_target_filtered, verbose=0), axis=1)
        cross_f1 = f1_score(y_target_filtered, y_pred_target, average="macro")
        cross_acc = accuracy_score(y_target_filtered, y_pred_target)
        cross_report = classification_report(
            y_target_filtered,
            y_pred_target,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
            output_dict=True,
        )

    model_path = output_dir / "step_d_encoder_attention_model.keras"
    model.save(model_path)

    results = {
        "step": "D",
        "model": "encoder_attention",
        "config": {
            "confidence_threshold": args.confidence_threshold,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patch_size": args.patch_size,
            "embedding_dim": args.embedding_dim,
            "num_heads": args.num_heads,
            "encoder_dim": args.encoder_dim,
            "dropout": args.dropout,
            "target_file": target_file.name,
        },
        "test_set": {
            "macro_f1": float(test_f1),
            "accuracy": float(test_acc),
        },
        "cross_dataset": {
            "macro_f1": float(cross_f1) if cross_f1 is not None else None,
            "accuracy": float(cross_acc) if cross_acc is not None else None,
            "classification_report": cross_report,
        },
        "training_history": {
            "epochs_trained": len(history.history.get("loss", [])),
            "final_loss": float(history.history["loss"][-1]) if history.history.get("loss") else None,
            "final_val_loss": float(history.history["val_loss"][-1]) if history.history.get("val_loss") else None,
        },
        "artifacts": {
            "model_file": str(model_path),
        },
    }

    results_path = output_dir / "step_d_encoder_attention_results.json"
    with open(results_path, "w") as file_obj:
        json.dump(results, file_obj, indent=2)

    print("\nResults:")
    print(f"  Test Macro-F1: {test_f1:.4f} | Accuracy: {test_acc:.4f}")
    if cross_f1 is not None:
        print(f"  Cross Macro-F1: {cross_f1:.4f} | Accuracy: {cross_acc:.4f}")
    print(f"  Saved model: {model_path}")
    print(f"  Saved report: {results_path}")


if __name__ == "__main__":
    main()
