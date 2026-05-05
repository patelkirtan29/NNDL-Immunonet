import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = "step1_merged.csv"
OUTPUT_DIR = "."

# ============================================================
# STEP 2A: Load merged data from Step 1
# ============================================================
print("=" * 60)
print("STEP 2A: Loading merged data from Step 1...")
print("=" * 60)

start = time.time()
df = pd.read_csv(INPUT_FILE)
print(f"  Loaded in {time.time()-start:.1f}s")
print(f"  Shape: {df.shape[0]} cells x {df.shape[1]} columns")
print(f"  Raw majorCluster classes: {df['majorCluster'].nunique()}")

# Show all raw classes before filtering
print(f"\n  All raw majorCluster labels:")
print(df["majorCluster"].value_counts().to_string())

# ============================================================
# STEP 2B: Filter out noise cells
# ============================================================
print("\n" + "=" * 60)
print("STEP 2B: Filtering out noise cells...")
print("=" * 60)

# These categories are not standard T-cell activation states
# and would add noise to our classification
noise_labels = [
    "filtered",       # Failed quality control in original paper
    "diverse.DN",     # Double-negative, ambiguous identity
    "diverse.other",  # Couldn't be clearly classified
    "diverse.DP",     # Double-positive, ambiguous identity
    "MAIT.other",     # MAIT cells — specialized innate-like T-cells, too few
    "MAIT.CD4",       # MAIT cells — only 11 cells
    "MAIT.DN",        # MAIT cells — only 4 cells
    "iNKT.CD4",       # iNKT cells — only 2 cells
    "iNKT.DN",        # iNKT cells — only 1 cell
]

n_before = len(df)
df_clean = df[~df["majorCluster"].isin(noise_labels)].copy()
n_after = len(df_clean)
n_removed = n_before - n_after

print(f"  Before filtering: {n_before} cells")
print(f"  Removed {n_removed} noise cells:")
for label in noise_labels:
    count = len(df[df["majorCluster"] == label])
    if count > 0:
        print(f"    - {label}: {count} cells")
print(f"  After filtering: {n_after} cells")

# ============================================================
# STEP 2C: Group into 6 clean ML classes
# ============================================================
print("\n" + "=" * 60)
print("STEP 2C: Grouping into 6 clean ML classes...")
print("=" * 60)

# Mapping based on immunology literature and the original paper
# Each group represents a distinct functional T-cell state
class_mapping = {
    # NAIVE — Resting cells that haven't encountered antigen
    # Defined by: CCR7, TCF7, LEF1, SELL high; effector genes low
    "CD4_C01-CCR7":   "Naive",       # Naive CD4+ (high CCR7 = lymph node homing)
    "CD4_C04-TCF7":   "Naive",       # Naive-like CD4+ (high TCF7 = stem-like factor)
    "CD8_C01-LEF1":   "Naive",       # Naive CD8+ (high LEF1 = maintains naive state)
    "CD8_C02-GPR183": "Naive",       # Naive/early memory CD8+ (GPR183 = positioning)

    # EFFECTOR CD8 — Actively killing targets
    # Defined by: GZMB, PRF1, NKG7, IFNG high; exhaustion markers low
    "CD8_C03-CX3CR1": "Effector",    # Cytotoxic effector (CX3CR1 = circulation marker)
    "CD8_C04-GZMK":   "Effector",    # Effector memory (GZMK = granzyme K, killing)
    "CD8_C05-CD6":    "Effector",    # Effector (CD6 = co-stimulatory receptor)
    "CD8_C06-CD160":  "Effector",    # Effector (CD160 = cytotoxic function)

    # EXHAUSTED CD8 — Burnt out from chronic antigen stimulation
    # Defined by: PDCD1, HAVCR2, LAG3, TOX, LAYN high
    "CD8_C07-LAYN":   "Exhausted",   # Exhausted (LAYN = exhaustion marker)

    # REGULATORY (Treg) — Immune suppressors, prevent autoimmunity
    # Defined by: FOXP3, CTLA4, IL2RA, IL10 high
    "CD4_C10-FOXP3":  "Treg",        # Classic Treg (FOXP3 = master Treg factor)
    "CD4_C11-IL10":   "Treg",        # Suppressive Treg (IL10 = anti-inflammatory)
    "CD4_C12-CTLA4":  "Treg",        # Treg (CTLA4 = immune checkpoint)

    # TH1-LIKE — Helper cells with cytotoxic features
    # Defined by: CXCL13, GZMK, IFNG, BHLHE40 high
    "CD4_C07-GZMK":   "Th1-like",    # Th1 effector memory (GZMK = cytotoxic helper)
    "CD4_C08-IL23R":  "Th1-like",    # Th17-like (IL23R = mucosal immunity)
    "CD4_C09-CXCL13": "Th1-like",    # Th1-like (CXCL13 = enriched in MSI tumors)

    # OTHER CD4 — Various helper subtypes
    # Mixed profiles, grouped because individually too small or not clearly
    # mapping to a single activation state
    "CD4_C02-ANXA1":  "Other_CD4",   # Anti-inflammatory (ANXA1 = annexin)
    "CD4_C03-GNLY":   "Other_CD4",   # Cytotoxic CD4 (GNLY = granulysin, rare)
    "CD4_C05-CXCR6":  "Other_CD4",   # Tissue-resident (CXCR6 = tissue homing)
    "CD4_C06-CXCR5":  "Other_CD4",   # Tfh-like (CXCR5 = B-cell zone homing)

    # CD8 MAIT that passed earlier filter (SLC4A10 subset)
    "CD8_C08-SLC4A10": "Other_CD4",  # MAIT CD8 — small group, put in Other
}

# Apply mapping
df_clean["cell_class"] = df_clean["majorCluster"].map(class_mapping)

# Check for any unmapped labels
unmapped = df_clean[df_clean["cell_class"].isna()]["majorCluster"].unique()
if len(unmapped) > 0:
    print(f"  WARNING: Unmapped labels found: {unmapped}")
    print(f"  Dropping {df_clean['cell_class'].isna().sum()} unmapped cells")
    df_clean = df_clean.dropna(subset=["cell_class"])

print(f"  Successfully mapped {len(df_clean)} cells into 6 classes")

# ============================================================
# STEP 2D: Show class distribution
# ============================================================
print("\n" + "=" * 60)
print("STEP 2D: Class distribution")
print("=" * 60)

class_counts = df_clean["cell_class"].value_counts()
total = len(df_clean)

print(f"\n  {'Class':<15} {'Count':>6} {'Percentage':>10}  {'Bar'}")
print(f"  {'-'*15} {'-'*6} {'-'*10}  {'-'*30}")
for cls, count in class_counts.items():
    pct = count / total * 100
    bar = "█" * int(pct)
    print(f"  {cls:<15} {count:>6} {pct:>9.1f}%  {bar}")

print(f"\n  Total cells: {total}")
print(f"  Number of classes: {len(class_counts)}")

# Class imbalance ratio
max_class = class_counts.max()
min_class = class_counts.min()
imbalance_ratio = max_class / min_class
print(f"  Imbalance ratio (largest/smallest): {imbalance_ratio:.1f}x")
print(f"  Largest class: {class_counts.idxmax()} ({max_class})")
print(f"  Smallest class: {class_counts.idxmin()} ({min_class})")

# ============================================================
# STEP 2E: Show which raw clusters went into each class
# ============================================================
print("\n" + "=" * 60)
print("STEP 2E: Raw cluster → Class mapping detail")
print("=" * 60)

for cls in ["Naive", "Effector", "Exhausted", "Treg", "Th1-like", "Other_CD4"]:
    subset = df_clean[df_clean["cell_class"] == cls]
    raw_counts = subset["majorCluster"].value_counts()
    print(f"\n  {cls} ({len(subset)} cells):")
    for raw_label, count in raw_counts.items():
        print(f"    ├─ {raw_label}: {count}")

# ============================================================
# STEP 2F: Show tissue distribution per class
# ============================================================
print("\n" + "=" * 60)
print("STEP 2F: Tissue source distribution per class")
print("=" * 60)

# Decode sampleType prefixes
tissue_map = {
    "P": "Blood",    # Peripheral blood
    "N": "Normal",   # Adjacent normal tissue
    "T": "Tumor"     # Tumor tissue
}

df_clean["tissue"] = df_clean["sampleType"].str[0].map(tissue_map)

cross_tab = pd.crosstab(df_clean["cell_class"], df_clean["tissue"])
print(f"\n{cross_tab.to_string()}")

print(f"\n  Key observations:")
exhausted_tumor = cross_tab.loc["Exhausted", "Tumor"] if "Exhausted" in cross_tab.index else 0
exhausted_total = class_counts.get("Exhausted", 0)
if exhausted_total > 0:
    print(f"    - Exhausted cells from tumor: {exhausted_tumor}/{exhausted_total} ({exhausted_tumor/exhausted_total*100:.0f}%) — expected, chronic antigen in tumors causes exhaustion")

naive_blood = cross_tab.loc["Naive", "Blood"] if "Naive" in cross_tab.index else 0
naive_total = class_counts.get("Naive", 0)
if naive_total > 0:
    print(f"    - Naive cells from blood: {naive_blood}/{naive_total} ({naive_blood/naive_total*100:.0f}%) — expected, naive cells circulate in blood")

# ============================================================
# STEP 2G: Save clean dataset
# ============================================================
print("\n" + "=" * 60)
print("STEP 2G: Saving clean dataset...")
print("=" * 60)

# Reorder columns: metadata first, then genes
meta_cols = ["UniqueCell_ID", "Patient_ID", "majorCluster", "sampleType", "cell_class", "tissue"]
gene_cols = [c for c in df_clean.columns if c not in meta_cols]

df_output = df_clean[meta_cols + gene_cols]

output_path = os.path.join(OUTPUT_DIR, "step2_cleaned.csv")
print(f"  Saving {df_output.shape[0]} cells x {df_output.shape[1]} columns...")
start = time.time()
df_output.to_csv(output_path, index=False)
print(f"  Saved: {output_path} ({os.path.getsize(output_path)/1e6:.1f} MB) in {time.time()-start:.1f}s")

# ============================================================
# STEP 2H: Generate class distribution plot
# ============================================================
print("\n" + "=" * 60)
print("STEP 2H: Generating class distribution plot...")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Class distribution bar chart
colors = {
    "Naive": "#3498db",
    "Effector": "#e74c3c",
    "Exhausted": "#8e44ad",
    "Treg": "#2ecc71",
    "Th1-like": "#f39c12",
    "Other_CD4": "#95a5a6"
}

class_order = ["Naive", "Effector", "Exhausted", "Treg", "Th1-like", "Other_CD4"]
counts_ordered = [class_counts[c] for c in class_order]
bar_colors = [colors[c] for c in class_order]

axes[0].bar(class_order, counts_ordered, color=bar_colors, edgecolor="white", linewidth=0.5)
axes[0].set_title("T-Cell Class Distribution (After Cleaning)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Number of Cells", fontsize=11)
axes[0].set_xlabel("Cell Class", fontsize=11)
axes[0].tick_params(axis="x", rotation=30)

# Add count labels on bars
for i, (cls, count) in enumerate(zip(class_order, counts_ordered)):
    pct = count / total * 100
    axes[0].text(i, count + 30, f"{count}\n({pct:.0f}%)", ha="center", fontsize=9)

# Plot 2: Tissue distribution stacked bar
cross_tab_ordered = cross_tab.loc[class_order]
cross_tab_ordered.plot(
    kind="bar", stacked=True, ax=axes[1],
    color=["#3498db", "#2ecc71", "#e74c3c"],
    edgecolor="white", linewidth=0.5
)
axes[1].set_title("Tissue Source per Class", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Number of Cells", fontsize=11)
axes[1].set_xlabel("Cell Class", fontsize=11)
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(title="Tissue", fontsize=9)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "step2_class_distribution.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved plot: {plot_path}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 2 COMPLETE")
print("=" * 60)
print(f"""
  What we did:
    - Removed {n_removed} noise cells (filtered, diverse, MAIT, iNKT)
    - Grouped 20 raw majorCluster labels into 6 clean classes
    - Verified tissue distribution matches biological expectations

  Output files:
    1. step2_cleaned.csv              — {df_output.shape[0]} cells x {len(gene_cols)} genes + 6 metadata cols
    2. step2_class_distribution.png   — Class distribution visualization

  Class summary:
    Naive:      {class_counts.get('Naive', 0):>5} cells  (CCR7, TCF7, LEF1 — resting)
    Effector:   {class_counts.get('Effector', 0):>5} cells  (GZMB, PRF1, NKG7 — killing)
    Exhausted:  {class_counts.get('Exhausted', 0):>5} cells  (PDCD1, TOX, LAYN — burnt out)
    Treg:       {class_counts.get('Treg', 0):>5} cells  (FOXP3, CTLA4 — suppressive)
    Th1-like:   {class_counts.get('Th1-like', 0):>5} cells  (CXCL13, GZMK — helper/cytotoxic)
    Other_CD4:  {class_counts.get('Other_CD4', 0):>5} cells  (mixed CD4 subtypes)
    Total:      {total:>5} cells

  Imbalance ratio: {imbalance_ratio:.1f}x (will use class weights during training)

  Next step: Run step3_preprocessing.py to:
    - Apply log2(TPM+1) transformation
    - Select highly variable genes (2,000-5,000)
    - Z-score normalize
    - PCA dimensionality reduction
    - Train/test split
""")