Step 0 — Baseline Experiments

Run the simple baselines to get quick sanity checks on source vs target performance.

Commands (from project root):

```bash
python experiments/step0_baselines/run_baselines.py
# or open and run the notebook
```

Outputs:
- `experiments/step0_baselines/results.csv` — per-model metrics
- `experiments/step0_baselines/models/` — saved models (joblib)

Next steps:
- Run CORAL alignment and re-run baselines
- Implement DANN in `experiments/step1_dann/`
