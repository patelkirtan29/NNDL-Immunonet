import pandas as pd
import numpy as np
import os
import time

# ============================================================
# CONFIGURATION — Update these paths to your local files
# ============================================================
TPM_FILE = "/Users/kirtan/Downloads/GSE108989_CRC.TCell.S11138.TPM.txt"
SOFT_FILE = "/Users/kirtan/Downloads/GSE108989_family.soft"
OUTPUT_DIR = "."  # saves in current directory

# ============================================================
# STEP 1A: Parse labels from SOFT file
# ============================================================
print("=" * 60)
print("STEP 1A: Parsing majorCluster labels from SOFT file...")
print("=" * 60)

start = time.time()

rows = []
header = None
with open(SOFT_FILE, "r", errors="replace") as f:
    for line in f:
        line = line.strip()

        # Find the header row
        if line.startswith("UniqueCell_ID"):
            header = [h.strip() for h in line.split("\t")]
            continue

        # After header, collect data rows
        if header and not line.startswith("!") and not line.startswith("^") and line:
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) >= 4:
                rows.append(parts[:4])

df_labels = pd.DataFrame(rows, columns=header[:4])

print(f"  Parsed {len(df_labels)} cell annotations in {time.time()-start:.1f}s")
print(f"  Columns: {list(df_labels.columns)}")
print(f"  Unique majorCluster values: {df_labels['majorCluster'].nunique()}")

# Save labels
labels_path = os.path.join(OUTPUT_DIR, "step1_labels.csv")
df_labels.to_csv(labels_path, index=False)
print(f"  Saved: {labels_path}")

# Show label distribution
print(f"\n  majorCluster distribution:")
print(df_labels["majorCluster"].value_counts().to_string())

# ============================================================
# STEP 1B: Load TPM expression matrix
# ============================================================
print("\n" + "=" * 60)
print("STEP 1B: Loading TPM expression matrix...")
print("  (This takes 1-2 minutes for the ~350MB file)")
print("=" * 60)

start = time.time()
df_raw = pd.read_csv(TPM_FILE, sep="\t", index_col=0)
print(f"  Loaded in {time.time()-start:.1f}s")
print(f"  Raw shape: {df_raw.shape[0]} genes x {df_raw.shape[1]} columns")

# ============================================================
# STEP 1C: Separate gene symbols and transpose
# ============================================================
print("\n" + "=" * 60)
print("STEP 1C: Transposing matrix (cells as rows, genes as columns)...")
print("=" * 60)

# First column is 'symbol' (gene names) — separate it
if df_raw.columns[0] == "symbol" or df_raw.iloc[:, 0].dtype == "object":
    gene_symbols = df_raw.iloc[:, 0].values   # gene name per row
    df_expr = df_raw.iloc[:, 1:]               # expression values only
    print(f"  Separated gene symbols column")
    print(f"  Expression matrix: {df_expr.shape[0]} genes x {df_expr.shape[1]} cells")
else:
    gene_symbols = df_raw.index.values
    df_expr = df_raw
    print(f"  No symbol column found, using index as gene IDs")

# Transpose: rows = cells, columns = genes
df_cells = df_expr.T
df_cells.columns = gene_symbols
df_cells.index.name = "UniqueCell_ID"

# Handle duplicate gene names (some genes map to multiple IDs)
# Keep first occurrence — standard practice
dup_genes = df_cells.columns[df_cells.columns.duplicated()]
if len(dup_genes) > 0:
    print(f"  Found {len(dup_genes)} duplicate gene names — keeping first occurrence")
    df_cells = df_cells.loc[:, ~df_cells.columns.duplicated()]

print(f"  Transposed shape: {df_cells.shape[0]} cells x {df_cells.shape[1]} genes")

# ============================================================
# STEP 1D: Merge expression data with labels
# ============================================================
print("\n" + "=" * 60)
print("STEP 1D: Merging expression data with labels...")
print("=" * 60)

# Reset index so UniqueCell_ID becomes a column
df_cells = df_cells.reset_index()

# Merge
df_merged = df_cells.merge(
    df_labels[["UniqueCell_ID", "Patient_ID", "majorCluster", "sampleType"]],
    on="UniqueCell_ID",
    how="inner"
)

# Count gene columns (everything except metadata)
meta_cols = ["UniqueCell_ID", "Patient_ID", "majorCluster", "sampleType"]
gene_cols = [c for c in df_merged.columns if c not in meta_cols]

print(f"  Merged shape: {df_merged.shape[0]} cells x {df_merged.shape[1]} columns")
print(f"  Gene columns: {len(gene_cols)}")
print(f"  Metadata columns: {meta_cols}")

# Check for any unmatched cells
n_labels = len(df_labels)
n_merged = len(df_merged)
if n_merged < n_labels:
    print(f"  Note: {n_labels - n_merged} cells in labels not found in TPM file")

# ============================================================
# STEP 1E: Quick data quality checks
# ============================================================
print("\n" + "=" * 60)
print("STEP 1E: Data quality checks...")
print("=" * 60)

# Check for NaN values
nan_count = df_merged[gene_cols].isna().sum().sum()
print(f"  NaN values in expression data: {nan_count}")

# Check sparsity
total_values = df_merged[gene_cols].shape[0] * df_merged[gene_cols].shape[1]
zero_values = (df_merged[gene_cols] == 0).sum().sum()
sparsity = zero_values / total_values * 100
print(f"  Sparsity: {sparsity:.1f}% zeros (normal for scRNA-seq)")

# Check data types
print(f"  Expression data type: {df_merged[gene_cols[0]].dtype}")
print(f"  Expression value range: [{df_merged[gene_cols].min().min():.2f}, {df_merged[gene_cols].max().max():.2f}]")

# ============================================================
# STEP 1F: Save outputs
# ============================================================
print("\n" + "=" * 60)
print("STEP 1F: Saving outputs...")
print("=" * 60)

# Save full merged dataset
merged_path = os.path.join(OUTPUT_DIR, "step1_merged.csv")
print(f"  Saving full merged dataset ({df_merged.shape[0]} cells x {df_merged.shape[1]} cols)...")
print(f"  This may take a minute due to file size...")
start = time.time()
df_merged.to_csv(merged_path, index=False)
print(f"  Saved: {merged_path} ({os.path.getsize(merged_path)/1e6:.1f} MB) in {time.time()-start:.1f}s")

# Save preview with key marker genes
key_genes = [
    "CCR7", "SELL", "TCF7", "LEF1",         # Naive
    "CD69", "IL2RA", "MKI67",                # Activation
    "GZMB", "GZMA", "PRF1", "NKG7",         # Effector
    "PDCD1", "HAVCR2", "LAG3", "TOX",        # Exhaustion
    "FOXP3", "CTLA4",                         # Treg
    "CXCL13", "IFNG", "CD4", "CD8A", "CD8B"  # General
]
available_genes = [g for g in key_genes if g in df_merged.columns]
preview_cols = ["UniqueCell_ID", "Patient_ID", "majorCluster", "sampleType"] + available_genes
df_preview = df_merged[preview_cols].head(10)

preview_path = os.path.join(OUTPUT_DIR, "step1_preview.csv")
df_preview.to_csv(preview_path, index=False)
print(f"  Saved: {preview_path}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)
print(f"""
  Files created:
    1. step1_labels.csv    — {len(df_labels)} cell labels with majorCluster
    2. step1_merged.csv    — {df_merged.shape[0]} cells x {len(gene_cols)} genes + labels
    3. step1_preview.csv   — 10 cells with key marker genes (open in Excel)

  Dataset summary:
    Cells:          {df_merged.shape[0]}
    Genes:          {len(gene_cols)}
    Classes:        {df_merged['majorCluster'].nunique()} unique majorCluster values
    Patients:       {df_merged['Patient_ID'].nunique()}
    Tissue sources: {df_merged['sampleType'].nunique()} types
    Sparsity:       {sparsity:.1f}%

  Key marker genes found: {len(available_genes)}/{len(key_genes)}
    Naive:      {[g for g in ['CCR7','SELL','TCF7','LEF1'] if g in available_genes]}
    Effector:   {[g for g in ['GZMB','GZMA','PRF1','NKG7'] if g in available_genes]}
    Exhaustion: {[g for g in ['PDCD1','HAVCR2','LAG3','TOX'] if g in available_genes]}
    Treg:       {[g for g in ['FOXP3','CTLA4'] if g in available_genes]}

  Preview (first 5 cells, key genes + label):
""")
print(df_preview[["UniqueCell_ID", "majorCluster"] + available_genes[:6]].head().to_string(index=False))

print(f"""
  Next step: Run step2_cleaning.py to:
    - Filter out noise cells (filtered, diverse, MAIT, iNKT)
    - Group 29 raw labels into 6 clean classes
    - Check class balance
""")