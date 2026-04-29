# ImmunoNet: Classifying T-Cell States from Gene Expression Using Deep Learning
**CSCI 4366/6366 — Neural Networks and Deep Learning | Fall 2026**  
Kirtan Patel · Swapnaneel Chatterjee

---

## Slide 1 — Problem (from proposal)

**Why this matters:** immunotherapy only works if exhausted T-cells can be reactivated.

- Flow cytometry measures only 15–30 surface proteins → frequent misclassification (e.g., PD-1 on both activated and exhausted cells)
- scRNA-seq captures ~20,000 genes per cell → full molecular fingerprint
- **ML goal:** classify T-cell states (Naive, Effector, Exhausted, Th1-like, Treg, Other CD4)
- **Hard requirement:** generalize across datasets from different sequencing technologies

---

## Slide 2 — Solution & Experiments (proposal baseline)

**Task:** multi-class supervised classification on GSE108989 → test cross-dataset on GSE126030.

- Data: GSE108989 (Smart-seq2, labeled) → GSE126030 (10x, unlabeled)
- Features: HVG selection + normalization + standardization
- Metrics (proposal): macro-F1, per-class F1, interpretability overlap with known markers
- Planned models (proposal): MLP, 1D-CNN, Self-Attention + classical baselines

---

## Slide 3 — What changed vs. proposal (explicit modifications)

**Key finding:** source-only models hit ~0.90 source F1 but collapsed on target.

**We rebuilt the project around domain adaptation and new experiments:**
- New experiments folder with **MMD Net**, **DANN v2**, **CDAN-E**, **Joint VAE**, and **GeneAttention series**
- Added preprocessing fixes: **Fano HVG selection**, **independent target standardization**, **dead gene removal**
- Goal shift: learn **domain-invariant** representations, not just better classifiers

---

## Slide 4 — Top 3 Architectures (highlight only)

**We’ll show the results PNGs live — this slide only highlights the top three.**

1. **GeneAtt v5 (best)** — preprocessing fixes + attention + alignment  
    Target F1: **0.438**
2. **GeneAtt v4** — 2-layer attention + mixup + entropy minimization  
    Target F1: **0.420**
3. **GeneAtt v3** — attention + MMD + centroid alignment  
    Target F1: **0.407**

---

## Slide 5 — Conclusions

- Proposal target met on source: **Source F1 ≈ 0.90 (≥ 0.80 goal)**
- Cross-dataset target still hard: **Target F1 = 0.438 (< 0.70 goal)**
- Biggest improvement came from **data fixes**, not architecture alone
- Takeaway: **domain shift + biology mismatch** is the main bottleneck

---

## Slide 6 — Future Work

- **Foundation models**: fine-tune Geneformer/scGPT for better cross-cancer transfer
- **Multi-source training**: add more cancer datasets (pan-cancer T-cell signatures)
- **Clinical translation**: test on patient biopsies to predict immunotherapy response

---

## Reference (not for slides)

**Other experiments run (keep for your notes):**
- Baselines: Logistic Regression (0.390), Random Forest (0.380), SVM (source 0.894)
- Domain adaptation: MMD Net (0.370), DANN v2 (0.385), CDAN-E (0.391), Joint VAE (0.395)
- GeneAtt v6 (0.392)

**Data sources:**
- GSE108989 (Zhang et al., Nature 2018)
- GSE126030 (Szabo et al., Nature Communications 2019)
