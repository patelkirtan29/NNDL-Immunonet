"""
Step B: Adversarial Domain Adaptation (DANN only)
=================================================
This script is independent from `domain_adaptive_models.py` and trains only DANN.
It does NOT train DA-SVM, Transformer, or GNN-SVM.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train only DANN for Step B")
    parser.add_argument("--confidence-threshold", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--domain-loss-weight", type=float, default=0.4)
    parser.add_argument("--grl-lambda", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--validation-split", type=float, default=0.1)
    return parser.parse_args()


def pick_gse126030_file(script_dir: Path) -> Path:
    correction_candidates = [
        script_dir.parent / "2_batch_correction/batch_correction_output/gse126030_corrected_coral.npy",
        script_dir.parent / "2_batch_correction/batch_correction_output/gse126030_corrected_combat.npy",
        script_dir.parent / "2_batch_correction/batch_correction_output/gse126030_corrected_mean_centered.npy",
        script_dir.parent / "2_batch_correction/batch_correction_output/gse126030_corrected_standardized.npy",
    ]
    selected = next((path for path in correction_candidates if path.exists()), None)
    return selected


def load_data(project_root: Path, script_dir: Path, confidence_threshold: float):
    X_train = np.load(project_root / "step3_X_train.npy").astype(np.float32)
    X_test = np.load(project_root / "step3_X_test.npy").astype(np.float32)
    y_train = np.load(project_root / "step3_y_train.npy")
    y_test = np.load(project_root / "step3_y_test.npy")

    with open(project_root / "step3_label_mapping.json") as file_obj:
        label_map = json.load(file_obj)
    class_names = [label_map[str(index)] for index in range(len(label_map))]

    selected_correction = pick_gse126030_file(script_dir)
    if selected_correction is not None:
        X_gse126030 = np.load(selected_correction).astype(np.float32)
    elif (project_root / "gse126030_preprocessed.npy").exists():
        X_gse126030 = np.load(project_root / "gse126030_preprocessed.npy").astype(np.float32)
        selected_correction = project_root / "gse126030_preprocessed.npy"
    else:
        raise FileNotFoundError("No GSE126030 input found")

    labels_path = script_dir.parent / "1_label_validation/label_validation_output/gse126030_reclustered_labels.csv"
    if not labels_path.exists():
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            class_names,
            selected_correction,
            None,
            None,
            None,
        )

    labels_df = pd.read_csv(labels_path)
    y_gse126030_names = labels_df["new_class"].values
    confidence = labels_df["confidence"].values

    use_mask = (confidence >= confidence_threshold) & (y_gse126030_names != "Uncertain")
    X_gse126030_filtered = X_gse126030[use_mask]
    y_gse126030_filtered = y_gse126030_names[use_mask]

    y_gse126030_numeric = np.array([class_names.index(name) for name in y_gse126030_filtered])

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        class_names,
        selected_correction,
        X_gse126030_filtered,
        y_gse126030_numeric,
        y_gse126030_filtered,
    )


def train_dann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_target: np.ndarray | None,
    y_target: np.ndarray | None,
    class_names: list[str],
    args: argparse.Namespace,
):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import callbacks, layers

    class GradientReversal(layers.Layer):
        def __init__(self, lambda_: float = 1.0, **kwargs):
            super().__init__(**kwargs)
            self.lambda_ = lambda_

        def call(self, inputs):
            lambda_ = self.lambda_

            @tf.custom_gradient
            def _flip_gradient(x):
                def grad(dy):
                    return -lambda_ * dy

                return x, grad

            return _flip_gradient(inputs)

    n_genes = X_train.shape[1]
    n_classes = len(class_names)

    target_for_dann = X_target if X_target is not None else X_test

    X_dann = np.vstack([X_train, target_for_dann]).astype(np.float32)
    y_class = np.concatenate([y_train, np.zeros(target_for_dann.shape[0], dtype=y_train.dtype)])
    y_domain = np.concatenate(
        [
            np.zeros(X_train.shape[0], dtype=np.int32),
            np.ones(target_for_dann.shape[0], dtype=np.int32),
        ]
    )

    class_sample_weight = np.concatenate(
        [
            np.ones(X_train.shape[0], dtype=np.float32),
            np.zeros(target_for_dann.shape[0], dtype=np.float32),
        ]
    )
    domain_sample_weight = np.ones(X_dann.shape[0], dtype=np.float32)

    gene_input = keras.Input(shape=(n_genes,), name="gene_input")
    shared = layers.Dense(256, activation="relu")(gene_input)
    shared = layers.Dropout(0.3)(shared)
    shared = layers.Dense(128, activation="relu")(shared)
    shared = layers.Dropout(0.2)(shared)

    class_hidden = layers.Dense(64, activation="relu")(shared)
    class_output = layers.Dense(n_classes, activation="softmax", name="class_output")(class_hidden)

    domain_shared = GradientReversal(lambda_=args.grl_lambda, name="grad_reverse")(shared)
    domain_hidden = layers.Dense(64, activation="relu")(domain_shared)
    domain_output = layers.Dense(2, activation="softmax", name="domain_output")(domain_hidden)

    model = keras.Model(inputs=gene_input, outputs=[class_output, domain_output], name="DANN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss={
            "class_output": "sparse_categorical_crossentropy",
            "domain_output": "sparse_categorical_crossentropy",
        },
        loss_weights={
            "class_output": 1.0,
            "domain_output": args.domain_loss_weight,
        },
        metrics={
            "class_output": ["accuracy"],
            "domain_output": ["accuracy"],
        },
    )

    history = model.fit(
        X_dann,
        [y_class, y_domain],
        sample_weight=[class_sample_weight, domain_sample_weight],
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        ],
        verbose=0,
    )

    y_pred_test = np.argmax(model.predict(X_test, verbose=0)[0], axis=1)
    test_f1 = f1_score(y_test, y_pred_test, average="macro")
    test_acc = accuracy_score(y_test, y_pred_test)

    cross_f1 = None
    cross_acc = None
    cross_report = None
    if X_target is not None and y_target is not None:
        y_pred_target = np.argmax(model.predict(X_target, verbose=0)[0], axis=1)
        cross_f1 = f1_score(y_target, y_pred_target, average="macro")
        cross_acc = accuracy_score(y_target, y_pred_target)
        cross_report = classification_report(
            y_target,
            y_pred_target,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
            output_dict=True,
        )

    return model, history, test_f1, test_acc, cross_f1, cross_acc, cross_report


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    output_dir = script_dir / "improved_models_output"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("STEP B: ADVERSARIAL DOMAIN ADAPTATION (DANN ONLY)")
    print("=" * 60)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        class_names,
        selected_correction,
        X_target,
        y_target,
        y_target_names,
    ) = load_data(project_root, script_dir, args.confidence_threshold)

    print("\nLoaded datasets:")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Target source: {selected_correction.name if selected_correction else 'N/A'}")
    if X_target is not None:
        print(f"  Target filtered: {X_target.shape} (threshold={args.confidence_threshold})")
        class_counts = pd.Series(y_target_names).value_counts()
        print("  Target class distribution:")
        for class_name, count in class_counts.items():
            print(f"    {class_name}: {count}")
    else:
        print("  Target filtered: unavailable (label file missing)")

    try:
        (
            model,
            history,
            test_f1,
            test_acc,
            cross_f1,
            cross_acc,
            cross_report,
        ) = train_dann(
            X_train,
            y_train,
            X_test,
            y_test,
            X_target,
            y_target,
            class_names,
            args,
        )
    except ImportError as error:
        print("\nTensorFlow is required for Step B DANN training.")
        print(f"ImportError: {error}")
        raise

    model_path = output_dir / "dann_model_step_b.keras"
    model.save(model_path)

    results = {
        "step": "B",
        "model": "dann",
        "config": {
            "confidence_threshold": args.confidence_threshold,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "domain_loss_weight": args.domain_loss_weight,
            "grl_lambda": args.grl_lambda,
            "learning_rate": args.learning_rate,
            "validation_split": args.validation_split,
            "target_correction_file": selected_correction.name if selected_correction else None,
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
        "training": {
            "epochs_trained": len(history.history.get("loss", [])),
            "final_loss": float(history.history["loss"][-1]) if history.history.get("loss") else None,
            "final_val_loss": float(history.history["val_loss"][-1]) if history.history.get("val_loss") else None,
        },
        "artifacts": {
            "model_file": str(model_path),
        },
    }

    results_path = output_dir / "step_b_dann_results.json"
    with open(results_path, "w") as file_obj:
        json.dump(results, file_obj, indent=2)

    print("\nResults:")
    print(f"  Test Macro-F1: {test_f1:.4f} | Accuracy: {test_acc:.4f}")
    if cross_f1 is not None and cross_acc is not None:
        print(f"  Cross Macro-F1: {cross_f1:.4f} | Accuracy: {cross_acc:.4f}")
    else:
        print("  Cross metrics: unavailable (no target labels)")

    print("\nSaved artifacts:")
    print(f"  Model: {model_path}")
    print(f"  Report: {results_path}")


if __name__ == "__main__":
    main()
