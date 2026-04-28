# ImmunoNet Improvements: Implementation Checklist

## Pre-Implementation

- [ ] Activate virtual environment
  ```bash
  cd /Users/kirtan/Projects/NNDL
  source .venv/bin/activate
  ```

- [ ] Verify all required training files exist
  ```bash
  ls -lh step3_*.npy step3_label_mapping.json
  ls -lh step2_cleaned.csv
  ```

- [ ] Verify GSE126030 preprocessed data exists
  ```bash
  ls -lh gse126030_preprocessed.npy  # From cross_dataset.py
  ```

- [ ] Install optional packages (if needed)
  ```bash
  pip install scanpy gseapy matplotlib seaborn
  ```

---

## Step 1: Label Validation

### Pre-execution
- [ ] Check input data
  ```bash
  cd improvements/1_label_validation
  # Verify it can read:
  python -c "import numpy as np; np.load('../../step3_X_train.npy'); print('✓')"
  ```

- [ ] Create output directory
  ```bash
  mkdir -p label_validation_output
  ```

### Run
- [ ] Execute script
  ```bash
  python gse126030_reclustering.py 2>&1 | tee label_validation.log
  ```

### Verify Output
- [ ] Check files created
  ```bash
  ls -lh label_validation_output/
  # Should have:
  # - gse126030_reclustered_labels.csv (100+ MB)
  # - reclustering_report.json
  # - reclustering_visualization.png
  ```

- [ ] Inspect CSV
  ```bash
  head -20 label_validation_output/gse126030_reclustered_labels.csv
  # Check columns: cell_id, original_condition, new_class, confidence
  ```

- [ ] Check confidence distribution
  ```bash
  python -c "import pandas as pd; df=pd.read_csv('label_validation_output/gse126030_reclustered_labels.csv'); print(f'Mean conf: {df[\"confidence\"].mean():.3f}, Min: {df[\"confidence\"].min():.3f}, Max: {df[\"confidence\"].max():.3f}')"
  ```

### Expected Results
```
Total cells: 63,877
High confidence (≥0.6): ~50,000
Naive cells predicted: ~40,000 (60%)
Effector cells: ~5,000 (8%)
Uncertain: ~10,000 (15%)
```

- [ ] If confidence too low (<0.5), adjust threshold in script line 120

---

## Step 2: Batch Correction

### Pre-execution
- [ ] Check input files
  ```bash
  cd ../2_batch_correction
  python -c "import numpy as np; print(np.load('../../step3_X_train.npy').shape)"  # Should be (n, 3000)
  ```

- [ ] Create output directory
  ```bash
  mkdir -p batch_correction_output
  ```

### Run
- [ ] Execute script
  ```bash
  python batch_effect_removal.py 2>&1 | tee batch_correction.log
  ```

### Verify Output
- [ ] Check files created
  ```bash
  ls -lh batch_correction_output/
  # Should have:
  # - gse108989_corrected_mean_centered.npy
  # - gse126030_corrected_mean_centered.npy
  # - batch_correction_comparison.png
  # - batch_correction_report.json
  ```

- [ ] Check batch separation improvement
  ```bash
  python -c "import json; r=json.load(open('batch_correction_output/batch_correction_report.json')); print('Batch separation scores:'); print('  Uncorrected: %.3f' % r['batch_separation_scores']['uncorrected']['batch_separation']); print('  Mean-centered: %.3f' % r['batch_separation_scores']['mean_centered']['batch_separation'])"
  ```

### Expected Results
```
Batch separation before: ~0.35-0.40
Batch separation after: ~0.10-0.15 (3-4x improvement)
Best method: mean_centered or standardized
```

- [ ] View visualization
  ```bash
  open batch_correction_output/batch_correction_comparison.png  # macOS
  # Should show PCA with red (GSE108989) and blue (GSE126030) mixed after correction
  ```

---

## Step 3: Improved Models

### Pre-execution
- [ ] Check batch-corrected files
  ```bash
  cd ../3_domain_adaptation
  ls -lh ../2_batch_correction/batch_correction_output/gse126030_corrected_mean_centered.npy
  ```

- [ ] Create output directory
  ```bash
  mkdir -p improved_models_output
  ```

### Run
- [ ] Execute script (WARNING: Takes 10-15 minutes)
  ```bash
  python domain_adaptive_models.py 2>&1 | tee improved_models.log
  ```

### Verify Output
- [ ] Check models saved
  ```bash
  ls -lh improved_models_output/
  # Should have:
  # - da_svm_model.pkl
  # - transformer_model.keras (if TensorFlow available)
  # - gnn_svm_model.pkl
  # - improved_models_results.json
  ```

- [ ] Check results JSON
  ```bash
  python -c "import json; r=json.load(open('improved_models_output/improved_models_results.json')); print('Cross-dataset F1 (expected 0.25-0.50):'); [print(f'  {m}: {v.get(\"macro_f1\", \"N/A\")}') for m, v in r['cross_dataset'].items()]"
  ```

### Expected Results
```
Test set F1 (should be ~0.88-0.89):
  Domain-adaptive SVM: 0.88-0.90
  Transformer: 0.87-0.89
  GNN-SVM: 0.88-0.90

Cross-dataset F1 (should improve from 0.08):
  Domain-adaptive SVM: 0.20-0.30
  Transformer: 0.25-0.40
  GNN-SVM: 0.30-0.50
```

- [ ] If cross-dataset F1 < 0.15:
  - Check batch correction (might need stricter correction)
  - Check re-clustered labels (confidence too low?)
  - Try adjusting hyperparameters in script

---

## Step 4: Statistical Validation

### Pre-execution
- [ ] Check all results folders exist
  ```bash
  cd ../4_statistical_validation
  ls -lh ../../results/{logistic_regression,svm_linear,mlp}/results.json
  ```

- [ ] Create output directory
  ```bash
  mkdir -p statistical_validation_output
  ```

### Run
- [ ] Execute script
  ```bash
  python statistical_tests.py 2>&1 | tee statistical_tests.log
  ```

### Verify Output
- [ ] Check results file
  ```bash
  ls -lh statistical_validation_output/statistical_tests_results.json
  ```

- [ ] Check Wilcoxon test results
  ```bash
  python -c "import json; r=json.load(open('statistical_validation_output/statistical_tests_results.json')); [print(f\"{k}: p={v.get('p_value', 'N/A'):.6f}, sig={v.get('significant_at_0.05', False)}\") for k,v in r.get('wilcoxon_tests', {}).items()]"
  ```

- [ ] Check Fisher's exact results (marker overlap)
  ```bash
  python -c "import json; r=json.load(open('statistical_validation_output/statistical_tests_results.json')); [[print(f\"  {cell_class}: {v.get('overlap_percentage', 0):.1f}%\") for cell_class, v in model_r.items()] for model_r in r.get('fishers_exact_tests', {}).values()]"
  ```

### Expected Results
```
Wilcoxon tests: p-values for model comparisons
Fisher's exact: Gene overlap significance
GSEA: Marker gene enrichment (ratio > 1.2)

Gene marker overlap (target ≥70%):
  Naive: ~50-60% (actual improvement might be lower)
  Effector: ~15-25% (can improve with better models)
  Exhausted: ~25-35%
```

---

## Step 5: Generate Report

### Pre-execution
- [ ] Check output directory
  ```bash
  cd ../6_results_report
  mkdir -p report_output
  ```

### Run
- [ ] Execute script
  ```bash
  python generate_report.py 2>&1 | tee report_generation.log
  ```

### Verify Output
- [ ] Check files created
  ```bash
  ls -lh report_output/
  # Should have:
  # - PROPOSAL_VS_RESULTS_REPORT.md
  # - report_summary.json
  ```

- [ ] Read the main report
  ```bash
  cat report_output/PROPOSAL_VS_RESULTS_REPORT.md | head -100
  ```

### Expected Content
Report should contain:
- [ ] Executive summary
- [ ] Success criteria scorecard (before/after)
- [ ] Root cause analysis
- [ ] Improvements implemented
- [ ] Expected results after improvements
- [ ] Recommendations for next steps (GNN, Transformer, etc.)

---

## Post-Implementation

### Data Backup
- [ ] Save key outputs to shared location
  ```bash
  cd improvements
  mkdir -p ../results_improvement_backup
  cp -r */*/output/* ../results_improvement_backup/ 2>/dev/null || true
  ```

### Create Summary
- [ ] Run summary script
  ```bash
  python << 'EOF'
import json
import os

print("\n" + "="*60)
print("IMPROVEMENTS SUMMARY")
print("="*60 + "\n")

folders = {
    "Label Validation": "1_label_validation/label_validation_output",
    "Batch Correction": "2_batch_correction/batch_correction_output",
    "Improved Models": "3_domain_adaptation/improved_models_output",
    "Statistical Tests": "4_statistical_validation/statistical_validation_output",
    "Report": "6_results_report/report_output"
}

for name, path in folders.items():
    files = os.listdir(path) if os.path.exists(path) else []
    print(f"✓ {name}: {len(files)} files")
    for f in sorted(files)[:3]:
        print(f"  - {f}")

print("\n" + "="*60)
print("📊 KEY METRICS TO CHECK:")
print("="*60)
print("1. Cross-dataset F1: Check improved_models_results.json")
print("2. Batch separation: Check batch_correction_report.json")
print("3. Gene overlap: Check statistical_tests_results.json")
print("4. Main findings: Read PROPOSAL_VS_RESULTS_REPORT.md")
print("\n")
EOF
  ```

### Documentation
- [ ] Update project README with results
- [ ] Add links to improvement outputs
- [ ] Document any hyperparameters used

---

## Troubleshooting

### Common Errors

#### "ModuleNotFoundError: No module named 'tensorflow'"
```bash
pip install tensorflow
# Or for specific version:
pip install tensorflow==2.12.0
```

#### "File not found: gse126030_preprocessed.npy"
```bash
# This file should be generated by cross_dataset.py
# If missing, run:
cd ..
python cross_dataset.py  # Saves gse126030_preprocessed.npy
```

#### "Confidence scores all near 0.5"
- Issue: kNN not discriminating between classes
- Fix: Check training data quality (step2_cleaned.csv not corrupted)
- Or: Increase k value in gse126030_reclustering.py line 87

#### "Batch correction shows no improvement"
- Issue: Batch effects might be real biological differences
- Try: ComBat method instead of mean-centering
- Or: Check if batch labels are correctly assigned

#### "Models train on test set but fail on cross-dataset"
- Issue: Domain shift too large
- Try: Stricter batch correction (ComBat + scVI)
- Or: Use higher domain weight (lambda) in DA-SVM

---

## Success Criteria for Each Step

| Step | Success Criteria | How to Check |
|------|-----------------|--------------|
| 1 | >50% cells high confidence | Check mean confidence in report |
| 2 | 3-4x reduction in batch separation | Compare before/after scores |
| 3 | Cross-dataset F1 ≥ 0.25 | Check improved_models_results.json |
| 4 | p<0.05 for model differences | Check wilcoxon_tests p-values |
| 5 | Report generated with recommendations | Read PROPOSAL_VS_RESULTS_REPORT.md |

---

## Final Sign-Off

- [ ] All 5 steps completed successfully
- [ ] No errors in logs
- [ ] Output files created and verified
- [ ] Cross-dataset F1 improved (check if ≥30%)
- [ ] Report generated and reviewed
- [ ] Ready for presentation to stakeholders

**Date Completed:** _________________  
**Completed By:** _________________  
**Notes:** _________________________________________________

---

**Next Phase:** After improvements are validated, proceed to:
1. Full GNN implementation (PyG/DGL)
2. Vision Transformer for genomics
3. Contrastive learning for domain alignment
