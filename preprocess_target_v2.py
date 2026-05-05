"""
ImmunoNet — Target Preprocessing v2 (GSE126030)
================================================
Fixes two critical issues in the original cross_dataset.py:

  Issue 1 — Wrong scaler:
    Original: applied source StandardScaler (fitted on Smart-seq2 TPM) to
              10x count data.  Result: target mean = -0.853, std = 0.118.
              CORAL then had to compensate an 8× variance mismatch.
    Fix:      standardise target INDEPENDENTLY (per-gene mean/std from target
              itself).  Both source and target now have std ≈ 1 before CORAL,
              making the covariance alignment numerically stable.

  Issue 2 — Dead genes:
    112 genes are essentially silent in 10x data (std < 0.05 in log space or
    >97% zeros).  They carry no signal for target prediction but add noise
    to distance metrics (MMD, centroid alignment, CORAL).
    Fix:      remove dead genes from BOTH source and target feature matrices
              and update step3_*.npy files so all experiment notebooks pick
              up the cleaner feature set automatically.

Run AFTER step3_v2.py has produced new step3_gene_names.npy.

Inputs:
  - gse126030_extracted/*.gz       (raw 10x count matrices)
  - step3_gene_names.npy           (source HVG list, 3000 genes)
  - step3_X_train.npy / step3_X_test.npy  (source feature matrices)

Outputs:
  - gse126030_preprocessed_v2.npy  (target, independently standardised)
  - step3_X_train.npy (overwritten — dead genes removed)
  - step3_X_test.npy  (overwritten — dead genes removed)
  - step3_gene_names.npy (overwritten — dead genes removed)
  - step3_gene_fano.npy  (overwritten — dead genes removed)
  - dead_gene_mask.npy   (boolean mask, True = kept)
  - target_scaler.pkl    (target StandardScaler for reference)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob
import os
import pickle
import json

# ─── Paths ────────────────────────────────────────────────────────────────────
EXTRACT_DIR = "gse126030_extracted"
OUTPUT_V2   = "gse126030_preprocessed_v2.npy"

# Thresholds for dead-gene detection (applied to target log-space values)
DEAD_STD_THRESHOLD  = 0.05   # per-gene std below this → dead
DEAD_ZERO_THRESHOLD = 0.97   # fraction of zero cells above this → dead

print("=" * 60)
print("Target Preprocessing v2 — GSE126030")
print("=" * 60)

# ─── 1. Load source gene list ─────────────────────────────────────────────────
print("\n[1] Loading source HVG list...")
train_gene_names = np.load("step3_gene_names.npy", allow_pickle=True)
print(f"    Source HVGs: {len(train_gene_names)}")

# ─── 2. Load raw 10x count matrices ───────────────────────────────────────────
print("\n[2] Loading raw 10x count matrices from", EXTRACT_DIR, "...")

def load_gz_matrix(filepath):
    """Load a GSE126030 .filtered.matrix.txt.gz file.
    Format: Row-0 = header (Accession, Gene, barcode1, barcode2, ...)
            Row-1+ = gene rows (Ensembl ID, symbol, counts...)
    Returns: (cells × genes matrix, gene_names list)
    """
    df       = pd.read_csv(filepath, sep="\t", compression="gzip")
    genes    = df["Gene"].values.tolist()
    matrix   = df.iloc[:, 2:].values.T.astype(np.float32)  # cells × genes
    return matrix, genes

gz_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, "*.gz")))
print(f"    Found {len(gz_files)} sample files")

all_matrices   = []
ref_gene_names = None

for fp in gz_files:
    sample = os.path.basename(fp).split("_")[0]
    mat, genes = load_gz_matrix(fp)
    print(f"    {sample}: {mat.shape[0]} cells × {mat.shape[1]} genes")

    if ref_gene_names is None:
        ref_gene_names = genes
    else:
        if genes != ref_gene_names:
            # Realign to first sample's gene order
            gene_to_idx = {g: i for i, g in enumerate(genes)}
            aligned = np.zeros((mat.shape[0], len(ref_gene_names)), dtype=np.float32)
            for i, g in enumerate(ref_gene_names):
                if g in gene_to_idx:
                    aligned[:, i] = mat[:, gene_to_idx[g]]
            mat = aligned

    all_matrices.append(mat)

X_cross = np.vstack(all_matrices)
print(f"\n    Combined: {X_cross.shape[0]} cells × {X_cross.shape[1]} genes")

# ─── 3. Gene alignment to source HVGs ────────────────────────────────────────
print("\n[3] Aligning to source HVGs...")

cross_gene_set = set(ref_gene_names)
train_gene_set = set(train_gene_names)
overlap        = cross_gene_set & train_gene_set
print(f"    Source HVGs: {len(train_gene_names)}")
print(f"    Target genes: {len(ref_gene_names)}")
print(f"    Overlap: {len(overlap)} ({len(overlap)/len(train_gene_names)*100:.1f}%)")

gene_to_idx_cross = {g: i for i, g in enumerate(ref_gene_names)}

X_aligned = np.zeros((X_cross.shape[0], len(train_gene_names)), dtype=np.float32)
n_matched = 0
for i, gene in enumerate(train_gene_names):
    if gene in gene_to_idx_cross:
        X_aligned[:, i] = X_cross[:, gene_to_idx_cross[gene]]
        n_matched += 1

n_zero_filled = len(train_gene_names) - n_matched
print(f"    Matched: {n_matched}  Zero-filled (missing): {n_zero_filled}")

# ─── 4. log2(count + 1) ───────────────────────────────────────────────────────
print("\n[4] Applying log2(count+1)...")
X_log = np.log2(X_aligned + 1)
print(f"    Range: [{X_log.min():.3f}, {X_log.max():.3f}]  mean={X_log.mean():.3f}")

# ─── 5. Dead-gene detection ────────────────────────────────────────────────────
print("\n[5] Detecting dead genes in target...")

tgt_std     = X_log.std(axis=0)
tgt_zero_fr = (X_aligned == 0).mean(axis=0)   # fraction of zero-count cells

dead_by_std  = tgt_std < DEAD_STD_THRESHOLD
dead_by_zero = tgt_zero_fr > DEAD_ZERO_THRESHOLD
dead_mask    = dead_by_std | dead_by_zero         # True = dead gene
alive_mask   = ~dead_mask                         # True = keep

n_dead = dead_mask.sum()
print(f"    std < {DEAD_STD_THRESHOLD}:           {dead_by_std.sum()} genes")
print(f"    zeros > {DEAD_ZERO_THRESHOLD*100:.0f}% of cells:  {dead_by_zero.sum()} genes")
print(f"    Total dead (either):   {n_dead} genes")
print(f"    Surviving genes:       {alive_mask.sum()}")

dead_gene_names = train_gene_names[dead_mask]
print(f"    Dead gene examples: {list(dead_gene_names[:15])}")

# ─── 6. Remove dead genes from target ─────────────────────────────────────────
print("\n[6] Removing dead genes from target...")
X_log_clean = X_log[:, alive_mask]
print(f"    Target shape after removal: {X_log_clean.shape}")

# ─── 7. Independent standardisation of target ────────────────────────────────
print("\n[7] Standardising target independently (per-gene mean/std from target)...")

tgt_scaler    = StandardScaler()
X_preprocessed = tgt_scaler.fit_transform(X_log_clean)
X_preprocessed = np.nan_to_num(X_preprocessed, nan=0.0).astype(np.float32)

print(f"    Target mean (should be ≈0): {X_preprocessed.mean():.6f}")
print(f"    Target std  (should be ≈1): {X_preprocessed.std():.4f}")
print(f"    Target per-gene std: min={X_preprocessed.std(axis=0).min():.4f} "
      f"median={np.median(X_preprocessed.std(axis=0)):.4f}")

# ─── 8. Update source matrices (remove dead genes) ────────────────────────────
print("\n[8] Updating source matrices (removing same dead genes)...")

X_train_old    = np.load("step3_X_train.npy").astype(np.float32)
X_test_old     = np.load("step3_X_test.npy").astype(np.float32)
gene_names_old = np.load("step3_gene_names.npy", allow_pickle=True)

X_train_clean  = X_train_old[:, alive_mask]
X_test_clean   = X_test_old[:,  alive_mask]
gene_names_clean = gene_names_old[alive_mask]

print(f"    Source train: {X_train_old.shape} → {X_train_clean.shape}")
print(f"    Source test:  {X_test_old.shape}  → {X_test_clean.shape}")

# Load fano scores if available
fano_path = "step3_gene_fano.npy"
if os.path.exists(fano_path):
    fano_old   = np.load(fano_path)
    fano_clean = fano_old[alive_mask]
    np.save(fano_path, fano_clean)
    print(f"    Fano scores updated: {fano_old.shape} → {fano_clean.shape}")

# ─── 9. Save everything ───────────────────────────────────────────────────────
print("\n[9] Saving outputs...")

np.save(OUTPUT_V2,                X_preprocessed)
np.save("step3_X_train.npy",      X_train_clean)
np.save("step3_X_test.npy",       X_test_clean)
np.save("step3_gene_names.npy",   gene_names_clean)
np.save("dead_gene_mask.npy",     alive_mask)

with open("target_scaler.pkl", "wb") as f:
    pickle.dump(tgt_scaler, f)

# Update config
config_path = "step3_config.json"
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    config["n_genes_after_dead_removal"] = int(alive_mask.sum())
    config["n_dead_genes_removed"]       = int(n_dead)
    config["dead_std_threshold"]         = DEAD_STD_THRESHOLD
    config["dead_zero_threshold"]        = DEAD_ZERO_THRESHOLD
    config["target_preprocessing"]       = "independent_standardisation_v2"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

print(f"    ✓ {OUTPUT_V2}                ({X_preprocessed.shape})")
print(f"    ✓ step3_X_train.npy           ({X_train_clean.shape})  ← overwritten")
print(f"    ✓ step3_X_test.npy            ({X_test_clean.shape})   ← overwritten")
print(f"    ✓ step3_gene_names.npy        ({gene_names_clean.shape}) ← overwritten")
print(f"    ✓ dead_gene_mask.npy          ({alive_mask.shape})")
print(f"    ✓ target_scaler.pkl")

# ─── 10. Diagnostics ──────────────────────────────────────────────────────────
print("\n[10] Diagnostics — before vs after fix:")

src_mean_old = X_train_old.mean()
src_std_old  = X_train_old.std(axis=0).mean()
tgt_mean_old = np.load("gse126030_preprocessed.npy").mean()
tgt_std_old  = np.load("gse126030_preprocessed.npy").std(axis=0).mean()

print(f"\n    BEFORE FIX:")
print(f"      Source global mean:        {src_mean_old:.4f}  (should be ≈0)")
print(f"      Target global mean:        {tgt_mean_old:.4f}  (was -0.85)")
print(f"      Source per-gene std (avg): {src_std_old:.4f}   (should be ≈1)")
print(f"      Target per-gene std (avg): {tgt_std_old:.4f}   (was 0.118 — 8× compressed)")

print(f"\n    AFTER FIX:")
print(f"      Source global mean:        {X_train_clean.mean():.4f}")
print(f"      Target global mean:        {X_preprocessed.mean():.4f}")
print(f"      Source per-gene std (avg): {X_train_clean.std(axis=0).mean():.4f}")
print(f"      Target per-gene std (avg): {X_preprocessed.std(axis=0).mean():.4f}")

print(f"""
{'='*60}
DONE.

Run order:
  1. python step3_v2.py              (source features with Fano HVG)
  2. python preprocess_target_v2.py  (this script — fixed target)
  3. Open and run experiments/step3_self_attention/05_gene_attention_v5.ipynb

The experiment notebooks load step3_X_train.npy and
gse126030_preprocessed_v2.npy automatically via:
  n_genes = X_train.shape[1]   # auto-adapts to new feature count
{'='*60}
""")
