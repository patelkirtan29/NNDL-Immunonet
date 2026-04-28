"""
Step D Architecture Visualization
================================
Creates a publication-friendly architecture diagram for the Step D
Encoder + Self-Attention classifier.

Run:
    python improvements/3_domain_adaptation/step_d_architecture_visualization.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _add_block(ax, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    block = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(block)
    ax.text(x + 0.015, y + h - 0.03, title, fontsize=11, fontweight="bold", va="top", ha="left", color="#111827")
    ax.text(x + 0.015, y + h - 0.065, body, fontsize=9, va="top", ha="left", color="#111827", linespacing=1.35)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color="#374151",
        )
    )


def draw_step_d_architecture(save_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Step D: Encoder + Self-Attention Classifier (Ethical, No Pseudo-Labeling)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    _add_block(
        ax,
        x=0.08,
        y=0.84,
        w=0.84,
        h=0.10,
        title="Input Layer",
        body="Gene expression vector: (batch, n_genes=20531)\nPreprocessing: source-fitted StandardScaler + optional CORAL-corrected target",
        color="#e0f2fe",
    )

    _add_block(
        ax,
        x=0.08,
        y=0.70,
        w=0.84,
        h=0.11,
        title="Encoder Stem",
        body="LayerNorm → Dense(128, ReLU) → Dropout(0.25)\nDense(64, ReLU) → Dropout(0.25)",
        color="#dcfce7",
    )

    _add_block(
        ax,
        x=0.08,
        y=0.53,
        w=0.84,
        h=0.14,
        title="Patch Tokenization",
        body="Truncate to n_tokens * patch_size\nReshape: (batch, n_tokens, patch_size=25)\nDense embedding: (batch, n_tokens, embedding_dim=64)\n+ Learnable positional embedding",
        color="#fef3c7",
    )

    _add_block(
        ax,
        x=0.08,
        y=0.31,
        w=0.84,
        h=0.18,
        title="Transformer Encoder Blocks × 2",
        body=(
            "Per block:\n"
            "• MultiHeadAttention(num_heads=4, key_dim=16) + Residual + LayerNorm\n"
            "• FeedForward: Dense(128, ReLU) → Dense(64) + Dropout\n"
            "• Residual + LayerNorm"
        ),
        color="#ede9fe",
    )

    _add_block(
        ax,
        x=0.08,
        y=0.18,
        w=0.84,
        h=0.10,
        title="Pooling + Classification Head",
        body="GlobalAveragePooling1D → Dense(128, ReLU) → Dropout(0.25) → Dense(6, Softmax)",
        color="#fee2e2",
    )

    _add_block(
        ax,
        x=0.08,
        y=0.05,
        w=0.84,
        h=0.10,
        title="Output + Ethical Evaluation",
        body="Predictions: 6 immune classes\nLoss: sparse categorical cross-entropy (source labels only)\nMetrics: Source test + cross-dataset (labeled subset only)",
        color="#f3f4f6",
    )

    _arrow(ax, (0.50, 0.84), (0.50, 0.81))
    _arrow(ax, (0.50, 0.70), (0.50, 0.67))
    _arrow(ax, (0.50, 0.53), (0.50, 0.49))
    _arrow(ax, (0.50, 0.31), (0.50, 0.28))
    _arrow(ax, (0.50, 0.18), (0.50, 0.15))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    svg_path = save_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    return save_path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    out_png = script_dir / "improved_models_output" / "step_d_architecture_diagram.png"
    saved = draw_step_d_architecture(out_png)
    print(f"Saved Step D diagram: {saved}")
    print(f"Saved Step D diagram (vector): {saved.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
