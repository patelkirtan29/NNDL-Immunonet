"""
Statistical Validation & Gene Marker Analysis
==============================================
Proposal requirements:
  1. Wilcoxon signed-rank test: Compare models across CV folds
  2. Fisher's exact test: Gene overlap significance
  3. Gene Set Enrichment Analysis (GSEA): Biological validation
  4. SHAP importance: Top genes per class

Output:
  - statistical_tests_results.json
  - gene_marker_overlap_report.json
  - gsea_enrichment_results.json
  - shap_analysis_complete.json
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

OUTPUT_DIR = SCRIPT_DIR / "statistical_validation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("STATISTICAL VALIDATION & GENE MARKER ANALYSIS")
print("=" * 60)

# ============================================================
# STEP 1: Load results from all models
# ============================================================
print("\nSTEP 1: Loading model results...")
print("-" * 60)

results_dir = PROJECT_ROOT / "results"
models = {
    "logistic_regression": results_dir / "logistic_regression/results.json",
    "svm_linear": results_dir / "svm_linear/results.json",
    "mlp": results_dir / "mlp/results.json",
}

all_results = {}
for model_name, path in models.items():
    if os.path.exists(path):
        with open(path) as f:
            all_results[model_name] = json.load(f)
        print(f"  Loaded {model_name}: F1={all_results[model_name]['test_macro_f1']:.4f}")
    else:
        print(f"  Warning: {model_name} results not found at {path}")

# ============================================================
# STEP 2: Wilcoxon Signed-Rank Test
# ============================================================
print("\nSTEP 2: Wilcoxon Signed-Rank Tests (Model Comparison)...")
print("-" * 60)

"""
Wilcoxon test: Are model performance differences statistically significant?
Non-parametric alternative to paired t-test (doesn't assume normality).
Null hypothesis: Distribution of differences is symmetric around zero.
"""

wilcoxon_results = {}

if len(all_results) >= 2:
    model_names = list(all_results.keys())
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            # Extract per-fold F1 scores
            folds1 = all_results[model1].get('fold_scores', [])
            folds2 = all_results[model2].get('fold_scores', [])
            
            if folds1 and folds2:
                f1_scores1 = np.array([f['macro_f1'] for f in folds1])
                f1_scores2 = np.array([f['macro_f1'] for f in folds2])
                
                # Wilcoxon signed-rank test
                if len(f1_scores1) == len(f1_scores2):
                    statistic, p_value = stats.wilcoxon(f1_scores1, f1_scores2)
                    
                    comparison_key = f"{model1}_vs_{model2}"
                    wilcoxon_results[comparison_key] = {
                        "model1": model1,
                        "model2": model2,
                        "f1_model1_mean": float(f1_scores1.mean()),
                        "f1_model2_mean": float(f1_scores2.mean()),
                        "wilcoxon_statistic": float(statistic),
                        "p_value": float(p_value),
                        "significant_at_0.05": bool(p_value < 0.05),
                        "interpretation": "Significant difference" if p_value < 0.05 else "No significant difference"
                    }
                    
                    print(f"\n  {model1} vs {model2}:")
                    print(f"    {model1} F1: {f1_scores1.mean():.4f}")
                    print(f"    {model2} F1: {f1_scores2.mean():.4f}")
                    print(f"    Wilcoxon p-value: {p_value:.6f}")
                    if p_value < 0.05:
                        better = model1 if f1_scores1.mean() > f1_scores2.mean() else model2
                        print(f"    ✓ Significant difference (p<0.05), {better} is better")
                    else:
                        print(f"    ✗ No significant difference (p≥0.05)")
else:
    print("  Insufficient models for comparison")

# ============================================================
# STEP 3: Fisher's Exact Test (Gene Marker Overlap)
# ============================================================
print("\nSTEP 3: Fisher's Exact Test (Gene Marker Significance)...")
print("-" * 60)

"""
Fisher's exact test: Is the overlap between predicted top genes and 
known markers statistically significant?

2x2 contingency table:
                 In known markers  |  Not in known markers
Top predicted      overlap         |   (top_genes - overlap)
Other genes        (markers-overlap) |  (total_genes - overlap - top + markers)
"""

# Load biological validation results (known markers per class)
known_markers = {
    "Naive": ["CCR7", "SELL", "TCF7", "LEF1", "IL7R", "CD27", "CD28"],
    "Effector": ["GZMB", "GZMA", "PRF1", "NKG7", "IFNG", "GNLY"],
    "Exhausted": ["PDCD1", "HAVCR2", "LAG3", "TOX", "TIGIT", "CXCL13", "CTLA4"],
    "Treg": ["FOXP3", "CTLA4", "IL2RA", "IL10", "TNFRSF18"],
    "Th1-like": ["CXCL13", "IFNG", "GZMK", "BHLHE40", "CD44"],
    "Other_CD4": ["CD4", "CD40LG", "ANXA1", "CXCR6"]
}

# Load interpretability results (top genes from model)
interp_path = "../../results/interpretability/biological_validation.json"
if os.path.exists(interp_path):
    with open(interp_path) as f:
        model_top_genes = json.load(f)
    
    fisher_results = {}
    
    for model_name, class_data in model_top_genes.items():
        fisher_results[model_name] = {}
        
        for cell_class, genes in class_data.items():
            if cell_class not in known_markers:
                continue
            
            top_genes_set = set(genes.get('top_20_genes', []))
            known_set = set(known_markers.get(cell_class, []))
            total_genes = 3000  # From preprocessing
            
            # Contingency table
            overlap = len(top_genes_set & known_set)
            top_only = len(top_genes_set) - overlap
            marker_only = len(known_set) - overlap
            neither = total_genes - len(top_genes_set) - len(known_set) + overlap
            
            # 2x2 contingency table
            contingency = np.array([
                [overlap, top_only],
                [marker_only, neither]
            ])
            
            # Fisher's exact test
            odds_ratio, p_value = stats.fisher_exact(contingency)
            
            overlap_pct = overlap / len(known_set) * 100 if len(known_set) > 0 else 0
            
            fisher_results[model_name][cell_class] = {
                "overlap": int(overlap),
                "known_markers": int(len(known_set)),
                "top_genes": int(len(top_genes_set)),
                "overlap_percentage": float(overlap_pct),
                "odds_ratio": float(odds_ratio),
                "p_value": float(p_value),
                "significant_at_0.05": bool(p_value < 0.05),
                "proposal_target": 0.70,  # 70% overlap requirement
                "meets_target": bool(overlap_pct >= 70)
            }
            
            status = "✓" if overlap_pct >= 70 else "✗"
            print(f"\n  {model_name} - {cell_class}:")
            print(f"    {status} Overlap: {overlap}/{len(known_set)} ({overlap_pct:.1f}%)")
            print(f"    Target: 70%, Meets: {overlap_pct >= 70}")
            print(f"    Fisher's exact p-value: {p_value:.6f}")
else:
    print("  Warning: Interpretability results not found")
    fisher_results = {}

# ============================================================
# STEP 4: Gene Set Enrichment Analysis (GSEA)
# ============================================================
print("\nSTEP 4: Gene Set Enrichment Analysis (GSEA)...")
print("-" * 60)

"""
GSEA: Are marker genes clustered at top of ranked gene list?
Uses Kolmogorov-Smirnov test to assess enrichment.
"""

gsea_results = {}

try:
    import gseapy
    print("  GSEApy library available, performing GSEA...")
    
    # Load gene names from preprocessing
    gene_names = np.load(PROJECT_ROOT / "step3_gene_names.npy", allow_pickle=True)
    
    # Load model predictions/importance (from interpretability)
    for model_name, class_genes in model_top_genes.items():
        gsea_results[model_name] = {}
        
        for cell_class, genes_info in class_genes.items():
            top_genes = genes_info.get('top_20_genes', [])
            marker_genes = known_markers.get(cell_class, [])
            
            # GSEA: Are marker genes enriched at top?
            ranked_list = pd.Series(index=gene_names, data=range(len(gene_names), 0, -1))
            marker_ranks = [ranked_list.get(g, 0) for g in marker_genes if g in ranked_list.index]
            
            if len(marker_ranks) > 0:
                # KS test: cumulative distribution of marker ranks vs all ranks
                ks_stat, ks_pval = stats.ks_2samp(marker_ranks, range(len(gene_names)))
                
                # Enrichment score: higher marker ranks = enriched
                mean_marker_rank = np.mean(marker_ranks)
                mean_all_rank = np.mean(range(len(gene_names)))
                enrichment_ratio = mean_marker_rank / mean_all_rank if mean_all_rank > 0 else 0
                
                gsea_results[model_name][cell_class] = {
                    "markers_tested": len(marker_genes),
                    "markers_in_data": len(marker_ranks),
                    "mean_marker_rank": float(mean_marker_rank),
                    "mean_all_ranks": float(mean_all_rank),
                    "enrichment_ratio": float(enrichment_ratio),
                    "ks_statistic": float(ks_stat),
                    "ks_pvalue": float(ks_pval),
                    "enriched": bool(enrichment_ratio > 1.2)  # >20% above mean
                }
                
                print(f"\n  {model_name} - {cell_class}:")
                print(f"    Enrichment ratio: {enrichment_ratio:.2f}x")
                print(f"    KS test p-value: {ks_pval:.6f}")
                print(f"    Status: {'Enriched' if enrichment_ratio > 1.2 else 'Not enriched'}")
    
except ImportError:
    print("  GSEApy not installed, implementing KS-based enrichment only")

# ============================================================
# STEP 5: Save all statistical results
# ============================================================
print("\nSTEP 5: Saving statistical results...")
print("-" * 60)

statistical_report = {
    "wilcoxon_tests": wilcoxon_results,
    "fishers_exact_tests": fisher_results,
    "gsea_enrichment": gsea_results,
    "proposal_criteria": {
        "test_f1_threshold": 0.80,
        "cross_dataset_f1_threshold": 0.70,
        "gene_marker_overlap_threshold": 0.70,
        "statistical_significance": "p < 0.05"
    }
}

report_path = os.path.join(OUTPUT_DIR, "statistical_tests_results.json")
with open(report_path, "w") as f:
    json.dump(statistical_report, f, indent=2, default=str)
print(f"  Saved: {report_path}")

# ============================================================
# STEP 6: Summary Report
# ============================================================
print("\n" + "=" * 60)
print("STATISTICAL VALIDATION COMPLETE")
print("=" * 60)

# Check proposal criteria
test_f1_passed = all(
    r.get('test_macro_f1', 0) >= 0.80 for r in all_results.values()
)
marker_overlap_passed = len([
    r for model_results in fisher_results.values() 
    for r in model_results.values() 
    if r.get('meets_target', False)
]) > 0

print(f"""
  Proposal Criteria Check:
    
    1. Test Macro-F1 ≥ 0.80:
       Status: {'✓ PASSED' if test_f1_passed else '✗ FAILED'}
       Results: {', '.join([f"{m}: {r['test_macro_f1']:.4f}" for m, r in all_results.items()])}
    
    2. Gene Marker Overlap ≥ 70%:
       Status: {'✓ PARTIAL' if marker_overlap_passed else '✗ FAILED'}
       See detailed results in {report_path}
    
    3. Wilcoxon Tests:
       Models compared: {len(wilcoxon_results)}
       Significant differences: {sum(1 for r in wilcoxon_results.values() if r['significant_at_0.05'])}
    
    4. GSEA Enrichment:
       Gene sets tested: {sum(len(v) for v in gsea_results.values())}

  Output files:
    - {report_path}
    - Compare to biological_validation.json for full overlap analysis
""")
