"""
Minimal baseline runner for Step 0 experiments.
Saves results to `experiments/step0_baselines/results.csv` and models to `experiments/step0_baselines/models/`.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
import joblib

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path.cwd()
EXP_DIR = PROJECT_ROOT / 'experiments' / 'step0_baselines'
MODELS_DIR = EXP_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data (same paths used across repo)
X_train = np.load(PROJECT_ROOT / 'step3_X_train.npy').astype(np.float32)
X_test = np.load(PROJECT_ROOT / 'step3_X_test.npy').astype(np.float32)
y_train = np.load(PROJECT_ROOT / 'step3_y_train.npy')
y_test = np.load(PROJECT_ROOT / 'step3_y_test.npy')

# Try to load optional target labeled subset
X_target_filtered = None
y_target_filtered = None
labels_csv = PROJECT_ROOT / 'improvements' / '1_label_validation' / 'label_validation_output' / 'gse126030_reclustered_labels.csv'
if labels_csv.exists():
    try:
        df = pd.read_csv(labels_csv)
        # If earlier notebook saved mask, attempt loading large target file
        target_path = PROJECT_ROOT / 'improvements' / '2_batch_correction' / 'batch_correction_output' / 'gse126030_corrected_coral.npy'
        if target_path.exists():
            X_target = np.load(target_path).astype(np.float32)
            # apply same confidence filtering as notebook if present
            if 'confidence' in df.columns and 'new_class' in df.columns:
                mask = (df['confidence'] >= 0.55) & (df['new_class'] != 'Uncertain')
                label_map = json.load(open(PROJECT_ROOT / 'step3_label_mapping.json'))
                class_names = [label_map[str(i)] for i in range(len(label_map))]
                X_target_filtered = X_target[mask.values]
                y_target_filtered = np.array([class_names.index(n) for n in df.loc[mask, 'new_class']], dtype=np.int64)
    except Exception:
        pass

results = []

# Logistic Regression baseline
print('Training Logistic Regression...')
clf_lr = LogisticRegression(max_iter=2000, random_state=SEED, n_jobs=-1)
clf_lr.fit(X_train, y_train)

pred_test = clf_lr.predict(X_test)
f1_test = f1_score(y_test, pred_test, average='macro')
acc_test = accuracy_score(y_test, pred_test)
print('LogReg Source macro-F1:', f1_test, 'Acc:', acc_test)

row = {'model':'LogisticRegression','domain':'source','f1_macro':float(f1_test),'accuracy':float(acc_test)}
results.append(row)
joblib.dump(clf_lr, MODELS_DIR / 'logreg.joblib')

if X_target_filtered is not None and y_target_filtered is not None:
    pred_target = clf_lr.predict(X_target_filtered)
    f1_target = f1_score(y_target_filtered, pred_target, average='macro')
    acc_target = accuracy_score(y_target_filtered, pred_target)
    print('LogReg Target macro-F1:', f1_target, 'Acc:', acc_target)
    results.append({'model':'LogisticRegression','domain':'target','f1_macro':float(f1_target),'accuracy':float(acc_target)})

# Random Forest baseline
print('\nTraining Random Forest...')
clf_rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
clf_rf.fit(X_train, y_train)

pred_test = clf_rf.predict(X_test)
f1_test = f1_score(y_test, pred_test, average='macro')
acc_test = accuracy_score(y_test, pred_test)
print('RF Source macro-F1:', f1_test, 'Acc:', acc_test)
results.append({'model':'RandomForest','domain':'source','f1_macro':float(f1_test),'accuracy':float(acc_test)})
joblib.dump(clf_rf, MODELS_DIR / 'random_forest.joblib')

if X_target_filtered is not None and y_target_filtered is not None:
    pred_target = clf_rf.predict(X_target_filtered)
    f1_target = f1_score(y_target_filtered, pred_target, average='macro')
    acc_target = accuracy_score(y_target_filtered, pred_target)
    print('RF Target macro-F1:', f1_target, 'Acc:', acc_target)
    results.append({'model':'RandomForest','domain':'target','f1_macro':float(f1_target),'accuracy':float(acc_target)})

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(EXP_DIR / 'results.csv', index=False)
print('\nSaved results to', EXP_DIR / 'results.csv')
print('Saved models to', MODELS_DIR)
