#!/usr/bin/env bash

# ImmunoNet Improvements: Quick Start Script
# ==========================================
# Runs all improvement scripts in sequence
# Total runtime: ~30-40 minutes

set -e  # Exit on any error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║      ImmunoNet Project: Run All Improvements               ║"
echo "║      Expected Runtime: 30-40 minutes                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python & packages
echo "📦 Checking dependencies..."
python -c "import numpy, pandas, sklearn; print('✓ Basic packages OK')" || {
    echo "❌ Missing packages. Run: pip install numpy pandas scikit-learn"
    exit 1
}

# Change to improvements directory
# cd improvements

echo ""
echo "═════════════════════════════════════════════════════════════"
echo "STEP 1: Label Validation & Re-clustering (GSE126030)"
echo "═════════════════════════════════════════════════════════════"
echo ""
echo "Purpose: Get proper 6-class labels instead of resting/activated"
echo "Time: ~5 minutes"
echo ""

python 1_label_validation/gse126030_reclustering.py
echo "✅ Step 1 complete!"
echo "   Output: 1_label_validation/label_validation_output/"
echo ""

echo "═════════════════════════════════════════════════════════════"
echo "STEP 2: Batch Effect Correction"
echo "═════════════════════════════════════════════════════════════"
echo ""
echo "Purpose: Remove systematic differences between GSE108989 & GSE126030"
echo "Time: ~5 minutes"
echo ""

python 2_batch_correction/batch_effect_removal.py
echo "✅ Step 2 complete!"
echo "   Output: 2_batch_correction/batch_correction_output/"
echo ""

echo "═════════════════════════════════════════════════════════════"
echo "STEP 3: Train Improved Models (DA-SVM, Transformer, GNN)"
echo "═════════════════════════════════════════════════════════════"
echo ""
echo "Purpose: Cross-dataset generalization with domain adaptation"
echo "Time: ~15 minutes"
echo ""

python 3_domain_adaptation/domain_adaptive_models.py
echo "✅ Step 3 complete!"
echo "   Output: 3_domain_adaptation/improved_models_output/"
echo ""

echo "═════════════════════════════════════════════════════════════"
echo "STEP 4: Statistical Validation (Wilcoxon, Fisher, GSEA)"
echo "═════════════════════════════════════════════════════════════"
echo ""
echo "Purpose: Test proposal criteria with statistical significance"
echo "Time: ~3 minutes"
echo ""

python 4_statistical_validation/statistical_tests.py
echo "✅ Step 4 complete!"
echo "   Output: 4_statistical_validation/statistical_validation_output/"
echo ""

echo "═════════════════════════════════════════════════════════════"
echo "STEP 5: Generate Comprehensive Report"
echo "═════════════════════════════════════════════════════════════"
echo ""
echo "Purpose: Compare proposal goals to achieved results"
echo "Time: ~2 minutes"
echo ""

python 6_results_report/generate_report.py
echo "✅ Step 5 complete!"
echo "   Output: 6_results_report/report_output/"
echo ""

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    ✅ ALL COMPLETE!                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Key Outputs:"
echo "───────────────────────────────────────────────────────────"
echo ""
echo "1️⃣  GSE126030 Re-clustered Labels"
echo "    File: 1_label_validation/label_validation_output/gse126030_reclustered_labels.csv"
echo "    What: Cell IDs with predicted class + confidence scores"
echo ""
echo "2️⃣  Batch Correction Analysis"
echo "    File: 2_batch_correction/batch_correction_output/batch_correction_comparison.png"
echo "    What: PCA visualization before/after batch correction"
echo ""
echo "3️⃣  Improved Model Results"
echo "    File: 3_domain_adaptation/improved_models_output/improved_models_results.json"
echo "    What: Test F1 & Cross-dataset F1 for DA-SVM, Transformer, GNN"
echo ""
echo "4️⃣  Statistical Test Results"
echo "    File: 4_statistical_validation/statistical_validation_output/statistical_tests_results.json"
echo "    What: Wilcoxon, Fisher's exact, GSEA p-values"
echo ""
echo "5️⃣  MAIN REPORT - Proposal vs Results"
echo "    File: 6_results_report/report_output/PROPOSAL_VS_RESULTS_REPORT.md"
echo "    What: Complete analysis with recommendations for next steps"
echo ""
echo "───────────────────────────────────────────────────────────"
echo ""
echo "📖 READ THE MAIN REPORT:"
echo "   cat 6_results_report/report_output/PROPOSAL_VS_RESULTS_REPORT.md"
echo ""
echo "🔍 EXPECTED IMPROVEMENTS:"
echo "   Cross-dataset F1: 0.08 → 0.30-0.50 (4-6x improvement)"
echo "   Batch separation: ~0.35 → ~0.10-0.15 (3-4x better)"
echo "   Gene marker overlap: 17-57% → 60-70%"
echo ""
echo "🚀 NEXT STEPS:"
echo "   1. Review PROPOSAL_VS_RESULTS_REPORT.md"
echo "   2. Check if targets met (cross-dataset F1 ≥ 30%?)"
echo "   3. If yes → Proceed to full GNN/Transformer architectures"
echo "   4. If no → Debug batch correction or label validation"
echo ""
echo "═════════════════════════════════════════════════════════════"
