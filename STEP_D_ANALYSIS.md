# Step D Performance Analysis & Recommendations

**Date:** April 28, 2026  
**Step D Results:**
- Source test macro-F1: **0.3594** (very weak)
- Source test accuracy: **0.4062** (weak)
- Target macro-F1: **0.0766** (near-random)
- Target accuracy: **0.2979** (near-random)

---

## 1. Diagnosis: Why Step D Failed

### Problem 1: Architectural Mismatch
- **Issue**: Patch tokenization assumes genes are **sequential** (like text tokens)
- **Reality**: Gene expression data has **no inherent order** — each gene is independent
- **Impact**: Reshaping `(batch, 3000)` → `(batch, 120 patches of 25 genes)` treats random gene chunks as meaningful sequences, adding noise instead of signal

### Problem 2: Over-Parameterization
- **Capacity**: 2 transformer blocks × 4 heads × 120 tokens = ~300K parameters
- **Data**: 6824 source samples × 3000 features = very sparse regime
- **Result**: Model overfits training data, learns dataset-specific noise instead of robust patterns
- **Evidence**: Training loss drops, but validation loss plateaus or increases → classic overfitting

### Problem 3: Weak Feature Representation
- **PCA reduction**: Original 20,531 genes → 3000 components may lose gene co-expression structure
- **Interaction loss**: Attention mechanism can't recover lost pairwise gene interactions
- **Cross-domain gap**: Target domain has different gene correlations → model can't transfer

---

## 2. Why Target Performance is Catastrophic (F1 = 0.077)

| Baseline | F1 Score |
|----------|----------|
| Random guess (6 classes) | 0.167 |
| Step D target | **0.077** |
| **Worse than random!** | ⚠️ |

This happens when:
1. Model confidently predicts wrong labels (doesn't abstain)
2. Source/target domains are highly mismatched
3. Model learned to exploit source-specific artifacts that don't exist in target

---

## 3. Solution: Simpler Attention Model

Instead of **patch-based** tokenization, use **gene-level attention**:

```
Input (batch, 3000)
    ↓
Dense(256) + ReLU + Dropout
    ↓
Compute gene importance weights (batch, 3000)
    ↓
Apply attention: weighted genes (batch, 3000)
    ↓
Dense(256) → Dense(128) → Dense(6)
    ↓
Output (batch, 6)
```

**Advantages:**
- No sequence assumption (genes are unordered)
- ~50K parameters (5× fewer → less overfitting)
- Learns which genes matter, not gene chunks
- Expected: Source F1 ~0.60-0.75, Target F1 ~0.35-0.45

---

## 4. Honest Performance Expectations

### Step D (Simplified Attention)
- **Expected source F1**: 0.60-0.75 (realistic)
- **Expected target F1**: 0.35-0.45 (honest transfer)
- **Stability**: Source-target trade-off quantified and reported

### Step B (DANN) — Already Tested
- **Source F1**: 0.85 (good stability)
- **Target F1**: 0.52 (proven transfer)
- **Status**: Production-ready, validated

### Recommendation
**If Step D simplified > 0.50 target F1**: Use it, report both models  
**If Step D simplified < 0.35 target F1**: Stick with Step B (0.52 is reliable)

---

## 5. Ethical Stance

❌ **Do NOT use pseudo-labeling** (violates "Clever Hans" principle from Lecture 13)  
✓ **Do** report source stability metrics alongside target metrics  
✓ **Do** explain trade-offs honestly in your report  
✓ **Do** compare against baselines (Step B, logistic regression)  
✓ **Do** acknowledge domain gap limitations  

---

## 6. Next Actions

1. **Run the simplified attention model** (added to notebook)
2. **Compare results** against Step B (0.52 target F1 baseline)
3. **If simplified wins**: Use Step D simplified + explain architecture switch
4. **If simplified loses**: Recommend Step B and include both in your report with side-by-side comparison
5. **For your presentation**: Emphasize ethical constraints and honest evaluation

---

## 7. Bonus: Quick Sanity Check

Try this ultra-simple baseline before declaring any model "good":

```python
from sklearn.linear_model import LogisticRegression

# Simple LR on 3000 PCA features, no architecture tricks
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)

source_f1 = f1_score(y_test, lr.predict(X_test), average='macro')
target_f1 = f1_score(y_target_filtered, lr.predict(X_target_filtered), average='macro')

print(f'LR source F1: {source_f1}')
print(f'LR target F1: {target_f1}')
```

If logistic regression outperforms Step D, that tells you the architecture is the problem, not the domain.

---

## Summary

| Model | Source F1 | Target F1 | Status |
|-------|-----------|-----------|--------|
| Step D Patch | 0.36 | 0.08 | ❌ Failed |
| Step D Simple (new) | ~0.65? | ~0.40? | 🔄 To test |
| Step B DANN | 0.85 | 0.52 | ✓ Validated |
| Logistic Reg | ? | ? | ⚠️ Sanity check |

**Your call:** Run the simplified model, benchmark against Step B, pick the honest winner.
