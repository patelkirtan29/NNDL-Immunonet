"""
Step C: Self-Training with Selective Pseudo-Labeling
====================================================
Independent self-training script that refines a classifier using target-domain
pseudo-labels in iterative rounds.

This script does NOT retrain the full Step A/Step B model suite.
It focuses on a single self-training loop that can be tuned quickly.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step C: iterative self-training")
    parser.add_argument("--confidence-threshold", type=float, default=0.55, help="Seed pseudo-label confidence threshold")
    parser.add_argument("--pseudo-threshold", type=float, default=0.70, help="Per-iteration pseudo-label confidence threshold")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum self-training iterations")
    parser.add_argument("--max-new-samples", type=int, default=4000, help="Maximum pseudo-labeled samples added per iteration")
    parser.add_argument("--per-class-cap", type=int, default=700, help="Maximum pseudo-labeled samples per class per iteration")
    parser.add_argument("--learning-rate", type=float, default=1.0, help="Inverse regularization strength C for LogisticRegression")
    parser.add_argument("--solver", type=str, default="lbfgs", help="LogisticRegression solver")
    parser.add_argument("--max-iter", type=int, default=2500, help="Maximum iterations for LogisticRegression")
    parser.add_argument("--use-seed-labels", action="store_true", default=True, help="Seed training with confident target labels from Step 1")
    parser.add_argument("--no-seed-labels", action="store_false", dest="use_seed_labels", help="Disable seed target labels")
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


def load_data(project_root: Path, script_dir: Path, seed_threshold: float, use_seed_labels: bool):
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
        seed_mask = (confidence >= seed_threshold) & (y_target_names != "Uncertain")
        X_seed = X_target[seed_mask] if use_seed_labels else np.empty((0, X_target.shape[1]), dtype=np.float32)
        y_seed_names = y_target_names[seed_mask] if use_seed_labels else np.array([], dtype=object)
        y_seed = np.array([class_names.index(name) for name in y_seed_names], dtype=np.int64)
    else:
        X_seed = np.empty((0, X_target.shape[1]), dtype=np.float32)
        y_seed = np.array([], dtype=np.int64)
        y_target_names = None
        seed_mask = np.zeros(X_target.shape[0], dtype=bool)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_target,
        class_names,
        target_file,
        X_seed,
        y_seed,
        y_target_names,
        seed_mask,
    )


def build_model(c_value: float, solver: str, max_iter: int) -> Pipeline:
    clf = LogisticRegression(
        C=c_value,
        solver=solver,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=42,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def select_pseudo_labels(
    probabilities: np.ndarray,
    remaining_indices: np.ndarray,
    max_new_samples: int,
    per_class_cap: int,
    confidence_threshold: float,
) -> np.ndarray:
    best_class = probabilities.argmax(axis=1)
    best_confidence = probabilities.max(axis=1)

    eligible = np.where(best_confidence >= confidence_threshold)[0]
    if eligible.size == 0:
        return np.array([], dtype=int)

    selected_positions: List[int] = []
    for class_idx in np.unique(best_class[eligible]):
        class_positions = eligible[best_class[eligible] == class_idx]
        order = np.argsort(-best_confidence[class_positions])
        capped = class_positions[order][:per_class_cap]
        selected_positions.extend(capped.tolist())

    if not selected_positions:
        return np.array([], dtype=int)

    selected_positions = np.array(selected_positions, dtype=int)
    order = np.argsort(-best_confidence[selected_positions])
    selected_positions = selected_positions[order][:max_new_samples]
    return remaining_indices[selected_positions]


def run_self_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_target: np.ndarray,
    class_names: List[str],
    args: argparse.Namespace,
    X_seed: np.ndarray,
    y_seed: np.ndarray,
    y_target_names: np.ndarray | None,
    seed_mask: np.ndarray,
):
    X_pool = X_target.copy()
    pool_indices = np.arange(X_pool.shape[0])

    if seed_mask.size and seed_mask.any():
        keep_mask = ~seed_mask
        X_pool = X_pool[keep_mask]
        pool_indices = pool_indices[keep_mask]

    train_X = X_train.copy()
    train_y = y_train.copy()

    if args.use_seed_labels and X_seed.shape[0] > 0:
        train_X = np.vstack([train_X, X_seed])
        train_y = np.concatenate([train_y, y_seed])
        if y_target_names is not None:
            used_seed_mask = np.zeros(X_pool.shape[0], dtype=bool)
            # Mark exact seed samples as removed by matching label validation mask via highest confidence subset.
            # This is intentionally conservative: the pool is still large enough for self-training.
            del used_seed_mask

    history: List[Dict[str, float]] = []
    model = None

    for iteration in range(1, args.max_iterations + 1):
        model = build_model(c_value=args.learning_rate, solver=args.solver, max_iter=args.max_iter)
        model.fit(train_X, train_y)

        test_pred = model.predict(X_test)
        test_f1 = f1_score(y_test, test_pred, average="macro")
        test_acc = accuracy_score(y_test, test_pred)

        probabilities = model.predict_proba(X_pool)
        selected_indices = select_pseudo_labels(
            probabilities=probabilities,
            remaining_indices=pool_indices,
            max_new_samples=args.max_new_samples,
            per_class_cap=args.per_class_cap,
            confidence_threshold=args.pseudo_threshold,
        )

        if selected_indices.size == 0:
            history.append({
                "iteration": iteration,
                "added_samples": 0,
                "train_size": int(train_X.shape[0]),
                "test_macro_f1": float(test_f1),
                "test_accuracy": float(test_acc),
            })
            break

        selected_pool_mask = np.isin(pool_indices, selected_indices)
        selected_pool_positions = np.where(selected_pool_mask)[0]
        selected_probs = probabilities[selected_pool_positions]
        selected_labels = selected_probs.argmax(axis=1)
        selected_confidences = selected_probs.max(axis=1)

        train_X = np.vstack([train_X, X_pool[selected_pool_positions]])
        train_y = np.concatenate([train_y, selected_labels])

        keep_mask = ~selected_pool_mask
        X_pool = X_pool[keep_mask]
        pool_indices = pool_indices[keep_mask]

        history.append({
            "iteration": iteration,
            "added_samples": int(selected_pool_positions.size),
            "train_size": int(train_X.shape[0]),
            "test_macro_f1": float(test_f1),
            "test_accuracy": float(test_acc),
            "mean_pseudo_confidence": float(selected_confidences.mean()) if selected_confidences.size else None,
            "selected_class_counts": {str(cls): int((selected_labels == cls).sum()) for cls in np.unique(selected_labels)},
        })

    final_model = model
    final_test_pred = final_model.predict(X_test)
    final_test_f1 = f1_score(y_test, final_test_pred, average="macro")
    final_test_acc = accuracy_score(y_test, final_test_pred)

    final_cross_f1 = None
    final_cross_acc = None
    final_cross_report = None
    if y_target_names is not None:
        target_eval_mask = y_target_names != "Uncertain"
        target_true = np.array([class_names.index(name) for name in y_target_names[target_eval_mask]], dtype=np.int64)
        final_target_pred = final_model.predict(X_target[target_eval_mask])
        final_cross_f1 = f1_score(target_true, final_target_pred, average="macro")
        final_cross_acc = accuracy_score(target_true, final_target_pred)
        final_cross_report = classification_report(
            target_true,
            final_target_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
            output_dict=True,
        )

    return final_model, history, final_test_f1, final_test_acc, final_cross_f1, final_cross_acc, final_cross_report


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_dir = script_dir / "improved_models_output"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("STEP C: SELF-TRAINING WITH SELECTIVE PSEUDO-LABELS")
    print("=" * 60)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_target,
        class_names,
        target_file,
        X_seed,
        y_seed,
        y_target_names,
        seed_mask,
    ) = load_data(project_root, script_dir, args.confidence_threshold, args.use_seed_labels)

    print("\nLoaded datasets:")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Target file: {target_file.name}")
    print(f"  Seed labels: {X_seed.shape[0]} samples (threshold={args.confidence_threshold})")
    print(f"  Pseudo-label threshold: {args.pseudo_threshold}")

    final_model, history, test_f1, test_acc, cross_f1, cross_acc, cross_report = run_self_training(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_target=X_target,
        class_names=class_names,
        args=args,
        X_seed=X_seed,
        y_seed=y_seed,
        y_target_names=y_target_names,
        seed_mask=seed_mask,
    )

    model_path = output_dir / "step_c_self_training_model.pkl"
    with open(model_path, "wb") as file_obj:
        pickle.dump(final_model, file_obj)

    results = {
        "step": "C",
        "model": "self_training",
        "config": {
            "confidence_threshold": args.confidence_threshold,
            "pseudo_threshold": args.pseudo_threshold,
            "max_iterations": args.max_iterations,
            "max_new_samples": args.max_new_samples,
            "per_class_cap": args.per_class_cap,
            "learning_rate": args.learning_rate,
            "solver": args.solver,
            "max_iter": args.max_iter,
            "use_seed_labels": args.use_seed_labels,
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
        "training_history": history,
        "artifacts": {
            "model_file": str(model_path),
        },
    }

    results_path = output_dir / "step_c_self_training_results.json"
    with open(results_path, "w") as file_obj:
        json.dump(results, file_obj, indent=2)

    print("\nResults:")
    print(f"  Test Macro-F1: {test_f1:.4f} | Accuracy: {test_acc:.4f}")
    if cross_f1 is not None and cross_acc is not None:
        print(f"  Cross Macro-F1: {cross_f1:.4f} | Accuracy: {cross_acc:.4f}")
    else:
        print("  Cross metrics: unavailable")

    print("\nIteration history:")
    for row in history:
        print(
            f"  Iter {row['iteration']}: added={row['added_samples']}, train_size={row['train_size']}, "
            f"test_f1={row['test_macro_f1']:.4f}, test_acc={row['test_accuracy']:.4f}"
        )

    print("\nSaved artifacts:")
    print(f"  Model: {model_path}")
    print(f"  Report: {results_path}")


if __name__ == "__main__":
    main()
