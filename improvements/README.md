# ImmunoNet Improvements: Complete Guide

## Overview

This folder contains all improvements to the ImmunoNet project, addressing gaps between the original proposal and current results.

**Main Goal:** Improve cross-dataset generalization from **8% to 30-50% F1** using:
1. Label validation & re-clustering
2. Batch effect correction
3. Domain-adaptive models (DA-SVM, Transformer, GNN)
4. Statistical validation (Wilcoxon, Fisher's exact, GSEA)

---

## Project Structure

```
improvements/
├── 1_label_validation/
│   ├── gse126030_reclustering.py          # Re-cluster GSE126030 with kNN
│   └── README.md
│
├── 2_batch_correction/
│   ├── batch_effect_removal.py            # ComBat/Harmony-like batch correction
│   └── README.md
│
├── 3_domain_adaptation/
│   ├── domain_adaptive_models.py          # DA-SVM, Transformer, GNN models
│   └── README.md
│
├── 4_statistical_validation/
│   ├── statistical_tests.py               # Wilcoxon, Fisher's exact, GSEA
│   └── README.md
│
├── 5_improved_models/                     # (Placeholder for outputs)
│   └── README.md
│
├── 6_results_report/
│   ├── generate_report.py                 # Proposal vs results analysis
│   └── PROPOSAL_VS_RESULTS_REPORT.md      # Main report (will be generated)
│
└── README.md                              # THIS FILE
```

---

## Quick Start (5 Minutes)

### Prerequisites

Ensure you're in the virtual environment:
```bash
cd /Users/kirtan/Projects/NNDL
source .venv/bin/activate
```

Check required packages:
```bash
pip list | grep -E "numpy|pandas|scikit-learn|tensorflow|matplotlib"
```

If missing:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn scipy
```

### Run Full Pipeline (25-40 minutes)

```bash
cd improvements

# 1. Validate GSE126030 labels (5 min)
echo "Step 1: Label Validation..."
python 1_label_validation/gse126030_reclustering.py

# 2. Correct batch effects (5 min)
echo "Step 2: Batch Correction..."
python 2_batch_correction/batch_effect_removal.py

# 3. Train improved models (15 min)
echo "Step 3: Improved Models..."
python 3_domain_adaptation/domain_adaptive_models.py

# 4. Statistical tests (3 min)
echo "Step 4: Statistical Validation..."
python 4_statistical_validation/statistical_tests.py

# 5. Generate report (2 min)
echo "Step 5: Generate Report..."
python 6_results_report/generate_report.py

echo "✅ All steps complete!"
```

---

## Detailed Steps

### Step 1: Label Validation (gse126030_reclustering.py)

**Purpose:** Re-cluster GSE126030 cells to get proper 6-class labels (instead of resting/activated)

**Input:**
- `step2_cleaned.csv` (training labels + class signatures)
- `step3_*.npy` (preprocessed training data)
- `gse126030_preprocessed.npy` (from cross_dataset.py)

**Output:**
- `1_label_validation/label_validation_output/gse126030_reclustered_labels.csv`
  - Columns: `cell_id`, `original_condition`, `new_class`, `confidence`
- `reclustering_report.json` (statistics)
- `reclustering_visualization.png` (confidence distribution)

**What it does:**
1. Trains kNN (k=15) on GSE108989 class signatures
2. Predicts class for all GSE126030 cells
3. Filters by confidence threshold (≥0.6)
4. Outputs mapping: "Resting" → which predicted classes?

**Expected Results:**
```
Resting cells (31,683):
  → Naive: ~70%
  → Other_CD4: ~20%
  → Uncertain: ~10%

Activated cells (32,194):
  → Effector: ~15%
  → Naive: ~50%  ← Should be higher if model works
  → Th1-like: ~5%
  → Uncertain: ~30%
```

**Run:**
```bash
cd 1_label_validation
python gse126030_reclustering.py
# Output files in: label_validation_output/
cd ..
```

---

### Step 2: Batch Effect Correction (batch_effect_removal.py)

**Purpose:** Remove systematic differences between GSE108989 and GSE126030

**Input:**
- `step3_X_train.npy`, `step3_X_test.npy` (GSE108989)
- `gse126030_preprocessed.npy` (GSE126030)

**Methods Implemented:**
1. **Mean Centering** - Shift each batch mean to overall mean
2. **Standardization** - Z-score per batch independently
3. **ComBat** - Full batch correction (if scanpy available)

**Output:**
- `2_batch_correction/batch_correction_output/`
  - `gse108989_corrected_mean_centered.npy`
  - `gse126030_corrected_mean_centered.npy`
  - `gse108989_corrected_standardized.npy`
  - `gse126030_corrected_standardized.npy`
  - `batch_correction_comparison.png` (before/after visualization)
  - `batch_correction_report.json` (batch separation scores)

**Quality Metric:** Batch Separation Score (0-1)
- **0** = batches completely mixed (ideal)
- **1** = batches completely separated (bad)
- **Expected before:** ~0.35-0.40
- **Expected after:** ~0.10-0.15 (3-4x improvement)

**Run:**
```bash
cd 2_batch_correction
python batch_effect_removal.py
# Output files in: batch_correction_output/
cd ..
```

**Key Insight:** PCA plot shows:
- **Before:** Red (GSE108989) and blue (GSE126030) clusters separate
- **After:** Red and blue mixed uniformly

---

### Step 3: Improved Models (domain_adaptive_models.py)

**Purpose:** Train models that generalize to new batches/datasets

**Input:**
- Training data + batch-corrected GSE126030
- Re-clustered GSE126030 labels (from Step 1)

**Models Trained:**

#### Model 1: Domain-Adaptive SVM
- **Idea:** Weight training instances by similarity to GSE126030 distribution
- **Method:** Train adversarial domain classifier → use probabilities as weights
- **Expected F1 on GSE126030:** 20-30% (vs 8% baseline)

#### Model 2: Transformer Classifier
- **Architecture:** 
  ```
  Input (3000 genes)
    → Dense projection (64 dim)
    → Add positional encoding
    → 2× Multi-head attention blocks (4 heads)
    → Global average pooling
    → Dense layer (64 units)
    → Output (6 classes)
  ```
- **Why:** Learns which gene combinations matter for each class
- **Expected F1:** ~88% (test), 25-40% (cross-dataset)

#### Model 3: GNN-SVM
- **Idea:** Create gene co-expression graph, use as features
- **Method:**
  1. Compute gene correlation matrix
  2. Create edges (r > 0.3)
  3. For each gene, add mean expression of co-expressed neighbors
  4. Train SVM on enriched features
- **Expected F1:** 30-50% (cross-dataset)

**Output:**
- `3_domain_adaptation/improved_models_output/`
  - `da_svm_model.pkl` (pickled SVM)
  - `transformer_model.keras` (TensorFlow model)
  - `gnn_svm_model.pkl` (SVM + graph)
  - `improved_models_results.json` (comparison table)

**Run:**
```bash
cd 3_domain_adaptation
python domain_adaptive_models.py
# Output files in: improved_models_output/
cd ..
```

### Step B Only: Adversarial Alignment (DANN-only, no SVM/Transformer/GNN retrain)

Use this when you want to iterate only on adversarial domain adaptation without running all models again.

**Script:** `3_domain_adaptation/step_b_adversarial_dann.py`

**Run:**
```bash
cd improvements
python 3_domain_adaptation/step_b_adversarial_dann.py
```

**Optional tuning run:**
```bash
cd improvements
python 3_domain_adaptation/step_b_adversarial_dann.py \
  --confidence-threshold 0.50 \
  --epochs 50 \
  --batch-size 128 \
  --domain-loss-weight 0.5 \
  --grl-lambda 0.6 \
  --learning-rate 5e-4
```

**Outputs:**
- `3_domain_adaptation/improved_models_output/dann_model_step_b.keras`
- `3_domain_adaptation/improved_models_output/step_b_dann_results.json`

### Step C Only: Self-Training Refinement

Use this to iterate on pseudo-labeling and improve cross-dataset metrics without retraining the Step A/Step B suite.

**Script:** `3_domain_adaptation/step_c_self_training.py`

**Run:**
```bash
cd improvements
python 3_domain_adaptation/step_c_self_training.py
```

**Recommended tuning run:**
```bash
cd improvements
python 3_domain_adaptation/step_c_self_training.py \
  --confidence-threshold 0.55 \
  --pseudo-threshold 0.70 \
  --max-iterations 3 \
  --max-new-samples 4000 \
  --per-class-cap 700
```

**Outputs:**
- `3_domain_adaptation/improved_models_output/step_c_self_training_model.pkl`
- `3_domain_adaptation/improved_models_output/step_c_self_training_results.json`

### Step D: Encoder + Self-Attention Classifier

Use this when you want to try a compact latent encoder and attention block for cross-dataset transfer.

**Script:** `3_domain_adaptation/step_d_encoder_attention.py`

**Run:**
```bash
cd improvements
python 3_domain_adaptation/step_d_encoder_attention.py
```

**Suggested tuning run:**
```bash
cd improvements
python 3_domain_adaptation/step_d_encoder_attention.py \
  --confidence-threshold 0.55 \
  --epochs 40 \
  --batch-size 128 \
  --learning-rate 2e-4 \
  --patch-size 25 \
  --embedding-dim 64 \
  --num-heads 4 \
  --encoder-dim 128
```

**Outputs:**
- `3_domain_adaptation/improved_models_output/step_d_encoder_attention_model.keras`
- `3_domain_adaptation/improved_models_output/step_d_encoder_attention_results.json`

**Key Output:** `improved_models_results.json`
```json
{
  "test_set": {
    "domain_adaptive_svm": {"macro_f1": 0.89, "accuracy": 0.90},
    "transformer": {"macro_f1": 0.88, "accuracy": 0.89},
    "gnn_svm": {"macro_f1": 0.89, "accuracy": 0.90}
  },
  "cross_dataset": {
    "domain_adaptive_svm": {"macro_f1": 0.25, "accuracy": 0.30},
    "transformer": {"macro_f1": 0.35, "accuracy": 0.42},
    "gnn_svm": {"macro_f1": 0.42, "accuracy": 0.48}
  }
}
```

---

### Step 4: Statistical Validation (statistical_tests.py)

**Purpose:** Rigorously test proposal criteria with statistical significance

**Tests Implemented:**

#### 1. Wilcoxon Signed-Rank Test
- **Compares:** Model performance across CV folds
- **Null hypothesis:** Distribution of differences = 0
- **Result:** p-value (sig if <0.05)
- **Example:** "SVM significantly better than Logistic Regression (p=0.031)"

#### 2. Fisher's Exact Test
- **Compares:** Top predicted genes vs known marker genes
- **Null hypothesis:** Overlap is random
- **Result:** Odds ratio + p-value
- **Example:** "Naive class genes significantly overlap with markers (p<0.001)"

#### 3. Gene Set Enrichment Analysis (GSEA)
- **Question:** Are marker genes ranked high in model importance?
- **Method:** KS test on ranked gene list
- **Result:** Enrichment ratio (>1.2 = significant)

**Output:**
- `4_statistical_validation/statistical_validation_output/`
  - `statistical_tests_results.json` (all p-values)
  - Shows Proposal Success Criteria scorecard

**Run:**
```bash
cd 4_statistical_validation
python statistical_tests.py
# Output files in: statistical_validation_output/
cd ..
```

**Key Output:** Proposal Scorecard
```
Proposal Criteria:
  1. Test F1 ≥ 0.80: ✅ PASSED (SVM: 0.8938)
  2. Cross-dataset F1 ≥ 0.70: ❌ FAILED (improved models: TBD)
  3. Gene marker overlap ≥ 70%: ⚠️ PARTIAL (Naive: 57%, Effector: 17%)
  4. Wilcoxon tests: ✅ DONE
  5. Fisher tests: ✅ DONE
```

---

### Step 5: Generate Comprehensive Report (generate_report.py)

**Purpose:** Compare original proposal to achieved results + recommendations

**Output:**
- `6_results_report/report_output/`
  - `PROPOSAL_VS_RESULTS_REPORT.md` (full markdown report)
  - `report_summary.json` (structured summary)

**What it includes:**
1. Success criteria scorecard (before/after)
2. Root cause analysis of failures
3. Improvements implemented
4. Expected outcomes
5. Recommendations for next steps

**Run:**
```bash
cd 6_results_report
python generate_report.py
# Output files in: report_output/
cd ..
```

**Read the full report:**
```bash
cat 6_results_report/report_output/PROPOSAL_VS_RESULTS_REPORT.md
```

---

## Expected Results Summary

### Before Improvements
| Metric | Value |
|--------|-------|
| Cross-dataset F1 | 0.08 (8%) |
| Batch separation | ~0.35 |
| Gene marker overlap (Naive) | 57% |
| Gene marker overlap (Effector) | 17% |

### After Improvements (Expected)
| Metric | Target | Method |
|--------|--------|--------|
| Cross-dataset F1 | 0.30-0.50 (30-50%) | DA-SVM + Transformer + GNN |
| Batch separation | 0.10-0.15 | Mean centering / ComBat |
| Gene marker overlap | 60-70% | Complete SHAP analysis |
| Statistical significance | p<0.05 | Wilcoxon + Fisher + GSEA |

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'scanpy'"
**Solution:** Install optional dependencies
```bash
pip install scanpy gseapy scVI-tools
```

### Error: "Cannot load GSE126030 preprocessed data"
**Solution:** Make sure cross_dataset.py was run and saved this file:
```bash
# From main directory, ensure this exists:
ls -lh gse126030_preprocessed.npy
# If not, run cross_dataset.py first
```

### Error: "TensorFlow not found"
**Solution:** Install TensorFlow
```bash
pip install tensorflow
# Or for GPU support:
pip install tensorflow[and-cuda]
```

### Models run too slowly
**Solution:** Reduce data or use GPU
```bash
# Check GPU availability:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# To limit to CPU only:
export CUDA_VISIBLE_DEVICES=-1
python script.py
```

---

## Output File Reference

### Key Files Generated

| File | Generated By | Contains |
|------|--------------|----------|
| `gse126030_reclustered_labels.csv` | Step 1 | New class labels for GSE126030 |
| `batch_correction_comparison.png` | Step 2 | PCA before/after batch correction |
| `improved_models_results.json` | Step 3 | Test F1, cross-dataset F1 per model |
| `statistical_tests_results.json` | Step 4 | P-values for all statistical tests |
| `PROPOSAL_VS_RESULTS_REPORT.md` | Step 5 | Comprehensive analysis & recommendations |

---

## Next Steps (After Improvements)

### If Cross-Dataset F1 < 30%
- Check batch correction effectiveness (Step 2)
- Verify GSE126030 labels are correct (Step 1)
- Try stricter confidence filtering in label validation

### If Cross-Dataset F1 ≥ 30%
- Implement full Graph Neural Network (not just SVM + adjacency)
- Try Vision Transformer (ViT) adapted for genomics
- Collect more GSE126030 validation data if available

### If Gene Marker Overlap Still < 70%
- Extract full SHAP explanations for each class
- Investigate Effector vs Exhausted confusion
- Consider hierarchical classification: first Naive/Activated, then subclasses

---

## References & Related Files

- **Original results:** `/Users/kirtan/Projects/NNDL/results/`
- **Training data:** `/Users/kirtan/Projects/NNDL/step2_cleaned.csv`
- **Preprocessing artifacts:** `/Users/kirtan/Projects/NNDL/step3_*.npy`
- **Original proposal:** (Share if available)

---

## Questions?

Common issues answered:
- "Why does DA-SVM weight instances?" → To emphasize cells similar to target domain (GSE126030)
- "Why re-cluster GSE126030?" → Original condition labels (resting/activated) don't match our 6 classes
- "Why not just use the SVM trained on GSE108989?" → It overfits to GSE108989-specific batch signatures
- "When to use which model?" → DA-SVM for speed, Transformer for interpretability, GNN for biology

---

**Last Updated:** April 2026  
**Status:** Ready for execution  
**Estimated Total Runtime:** 30-40 minutes
