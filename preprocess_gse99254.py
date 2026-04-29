"""
GSE99254 Preprocessing — NSCLC T-Cell Dataset (Guo et al., Nature Medicine 2018)
=================================================================================
Target dataset for cross-dataset generalisation evaluation.

Key advantage over GSE126030:
  - Same technology: Smart-seq2 TPM (identical to source GSE108989)
  - No technology mismatch → no CORAL alignment needed
  - Ground-truth cluster labels derivable from cell IDs + coarse mapping

Cell ID format: [tissue][cell_type][number]-[date]
  Tissue prefix: NT=Normal adjacent tissue, PT=Peripheral blood, TT=Tumor
  Type letter:   C=CD8+, H=CD4+ helper, R=Regulatory(Treg), Y=other CD4, S=naive/other

Label strategy:
  Step 1 — coarse labels from cell ID (5 classes, R→Treg is reliable)
  Step 2 — GeneAttentionV5 predictions on processed data as pseudo-labels
            (high confidence expected since technology matches source exactly)

Inputs:
  - /Users/kirtan/Downloads/GSE99254_NSCLC.TCell.S12346.TPM.txt
  - step3_gene_names.npy   (2701 source HVGs after dead-gene removal)
  - step3_X_train.npy      (for source scaler fitting reference)

Outputs:
  - gse99254_preprocessed.npy      (cells × 2701 genes, standardised)
  - gse99254_cell_meta.csv         (cell_id, tissue, coarse_type, patient)
  - gse99254_coarse_labels.npy     (integer labels from cell ID parsing)
  - gse99254_label_mapping.json    (coarse int → class name)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os, json, re

# ── Paths ─────────────────────────────────────────────────────────────────────
TPM_FILE   = "/Users/kirtan/Downloads/GSE99254_NSCLC.TCell.S12346.TPM.txt"
ROOT       = "/Users/kirtan/Projects /NNDL"
OUT_MATRIX = os.path.join(ROOT, "gse99254_preprocessed.npy")
OUT_META   = os.path.join(ROOT, "gse99254_cell_meta.csv")
OUT_LABELS = os.path.join(ROOT, "gse99254_coarse_labels.npy")
OUT_LMAP   = os.path.join(ROOT, "gse99254_label_mapping.json")

print("=" * 60)
print("GSE99254 Preprocessing — NSCLC T-Cell Dataset")
print("=" * 60)

# ── 1. Load source HVG list ────────────────────────────────────────────────────
print("\n[1] Loading source HVG list...")
gene_names = np.load(os.path.join(ROOT, "step3_gene_names.npy"), allow_pickle=True)
n_genes    = len(gene_names)
print(f"    Source HVGs: {n_genes}")

# ── 2. Load TPM matrix ────────────────────────────────────────────────────────
print("\n[2] Loading GSE99254 TPM matrix...")
df = pd.read_csv(TPM_FILE, sep="\t", index_col=None)
print(f"    Raw shape: {df.shape[0]} genes × {df.shape[1]-2} cells")

tgt_gene_names = df["symbol"].values
cell_ids       = df.columns[2:].tolist()   # skip geneID, symbol
X_raw          = df.iloc[:, 2:].values.T.astype(np.float32)  # cells × genes
print(f"    Matrix shape: {X_raw.shape}")

# ── 3. log2(TPM + 1) ─────────────────────────────────────────────────────────
print("\n[3] Applying log2(TPM+1)...")
X_log = np.log2(X_raw + 1.0)
print(f"    Range: [{X_log.min():.3f}, {X_log.max():.3f}]  mean={X_log.mean():.3f}")

# ── 4. Align to source HVGs ───────────────────────────────────────────────────
print("\n[4] Aligning to source HVGs...")
gene_to_idx = {g: i for i, g in enumerate(tgt_gene_names)}
X_aligned   = np.zeros((len(cell_ids), n_genes), dtype=np.float32)
n_matched   = 0
for j, gene in enumerate(gene_names):
    if gene in gene_to_idx:
        X_aligned[:, j] = X_log[:, gene_to_idx[gene]]
        n_matched += 1

n_missing = n_genes - n_matched
print(f"    Matched: {n_matched} / {n_genes}  ({n_matched/n_genes*100:.1f}%)")
print(f"    Zero-filled (missing): {n_missing}")

# ── 5. Independent standardisation ───────────────────────────────────────────
print("\n[5] Standardising independently (per-gene mean/std from GSE99254)...")
scaler       = StandardScaler()
X_processed  = scaler.fit_transform(X_aligned)
X_processed  = np.nan_to_num(X_processed, nan=0.0).astype(np.float32)
print(f"    Mean (should be ≈0): {X_processed.mean():.6f}")
print(f"    Std  (should be ≈1): {X_processed.std():.4f}")

# ── 6. Parse cell ID metadata ─────────────────────────────────────────────────
print("\n[6] Parsing cell ID metadata...")

TISSUE_MAP = {"NT": "Normal_adjacent", "PT": "Peripheral_blood",
              "TT": "Tumor", "T-": "Tumor_other"}

# Cell type letter → coarse T-cell class
# Based on Guo et al. 2018 cluster structure:
#   C = CD8+ cytotoxic (Effector / Exhausted)
#   H = CD4+ helper    (Other_CD4 / Th1-like)
#   R = Regulatory     (Treg)
#   Y = CD4+ other     (Naive / Other_CD4)
#   S = Naive-like     (Naive)
TYPE_MAP = {
    "C": "CD8_T",       # CD8+ — will be split by model into Effector/Exhausted/Th1-like
    "H": "CD4_helper",  # CD4+ helper — Other_CD4 / Th1-like
    "R": "Treg",        # Regulatory — most reliable label
    "Y": "CD4_other",   # Other CD4
    "S": "Naive_like",  # Naive-like
}

# Coarse label mapping (5 classes from cell ID)
COARSE_CLASSES = ["CD8_T", "CD4_helper", "Treg", "CD4_other", "Naive_like"]
coarse_to_int  = {c: i for i, c in enumerate(COARSE_CLASSES)}

rows = []
coarse_labels = []
for cid in cell_ids:
    # Parse tissue
    if cid[:2] in TISSUE_MAP:
        tissue    = TISSUE_MAP[cid[:2]]
        type_char = cid[2] if len(cid) > 2 else "?"
        patient   = re.sub(r'[A-Z]+-(\d+)-.*', r'\1', cid)
    elif cid[:2] == "T-":
        tissue    = TISSUE_MAP["T-"]
        type_char = cid[2] if len(cid) > 2 else "?"
        patient   = "unknown"
    else:
        tissue    = "unknown"
        type_char = "?"
        patient   = "unknown"

    coarse = TYPE_MAP.get(type_char, "CD8_T")
    rows.append({"cell_id": cid, "tissue": tissue,
                 "type_char": type_char, "coarse_type": coarse,
                 "patient": patient})
    coarse_labels.append(coarse_to_int[coarse])

meta_df      = pd.DataFrame(rows)
coarse_arr   = np.array(coarse_labels, dtype=np.int64)

print(f"    Tissue distribution:")
print(meta_df["tissue"].value_counts().to_string())
print(f"\n    Coarse type distribution:")
print(meta_df["coarse_type"].value_counts().to_string())

# ── 7. Save outputs ───────────────────────────────────────────────────────────
print("\n[7] Saving outputs...")
np.save(OUT_MATRIX, X_processed)
np.save(OUT_LABELS, coarse_arr)
meta_df.to_csv(OUT_META, index=False)
with open(OUT_LMAP, "w") as f:
    json.dump({str(v): k for k, v in coarse_to_int.items()}, f, indent=2)

print(f"    ✓ {OUT_MATRIX}      {X_processed.shape}")
print(f"    ✓ {OUT_META}")
print(f"    ✓ {OUT_LABELS}")
print(f"    ✓ {OUT_LMAP}")

# ── 8. Diagnostics ────────────────────────────────────────────────────────────
print("\n[8] Diagnostics vs source dataset...")
X_train = np.load(os.path.join(ROOT, "step3_X_train.npy")).astype(np.float32)
print(f"    Source  mean: {X_train.mean():.4f}   std/gene: {X_train.std(0).mean():.4f}")
print(f"    Target  mean: {X_processed.mean():.4f}   std/gene: {X_processed.std(0).mean():.4f}")
print(f"    Mean diff (abs): {np.abs(X_train.mean(0) - X_processed.mean(0)).mean():.4f}")
print()
print("    NOTE: No CORAL alignment applied — same technology (Smart-seq2 TPM).")
print("    Mean diff should be much lower than GSE126030 (was 0+ even after CORAL).")
print()
print("=" * 60)
print("DONE. Next: run 05_gene_attention_v5.ipynb pointing to gse99254_preprocessed.npy")
print("      and gse99254_cell_meta.csv for evaluation.")
print("=" * 60)
