# Benign vs. Malignant Tumor Classifier Using Logistic Regression

> **Educational demo only — not for clinical use.**  
> Predicts **benign (1)** vs **malignant (0)** from 30 fine-needle aspirate (FNA) cytology features with a simple, transparent pipeline.

---

## Overview

- **Goal:** Clean, reproducible baseline for tumor classification on tabular FNA features.  
- **Pipeline:** `StandardScaler → LogisticRegression (L2, class_weight='balanced')`  
- **Evaluation:** Proper **held-out test split** (no leakage) with Accuracy, ROC AUC, Confusion Matrix, and per-class metrics.  
- **Artifacts:**  
  - `tumor_model.joblib` — serialized pipeline (scaler + model)  
  - `heldout_test.json` — all **non-training (held-out)** samples with features, ground truth, predictions, and probabilities  
  - Optional **PyQt5 UI** to browse held-out samples or manually enter 30 features and predict

---

## Data

- **Source:** scikit-learn’s *Breast Cancer Wisconsin (Diagnostic)* (WDBC).  
- **Shape:** 569 samples × 30 real-valued features (e.g., radius, texture, perimeter, area; each summarized as mean / standard error / worst).  
- **Labels:** `0 = malignant`, `1 = benign`.  
- **Imbalance:** Mild (≈63% benign, ≈37% malignant).

---

## Preprocessing & Splits

- **Held-out test set:** `train_test_split(..., stratify=y, random_state=42)` with a default **20%** test fraction.  
- **Standardization:** `StandardScaler` is **fit on the training split only** and applied to validation/test/new data.  
  - Intuition: For each feature \(x_j\), transform to \(z_j = (x_j - \mu_j)/\sigma_j\) for mean≈0 and variance≈1 → stable optimization, fair L2 penalty, and interpretable coefficients.
- **Leakage prevention:** Cross-validation (if used) runs **only on the training split**.

**Produced files**
- `tumor_model.joblib` — saved model bundle with `feature_names` and `target_names`.  
- `heldout_test.json` — records for every held-out sample (original index, true label, predicted label, probabilities, full 30-feature dict).

---

## Model

**Logistic Regression with L2 regularization** (scikit-learn):

- **Score & probability**
  \[
  z = w^\top x + b, \quad p(y=1\mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}
  \]
- **Objective (regularized log loss)**
  \[
  \min_{w,b}\; \sum_i \big[-y_i\log p_i - (1-y_i)\log(1-p_i)\big] \;+\; \lambda \lVert w\rVert_2^2
  \]
  where \(C = 1/\lambda\) in scikit-learn (default \(C=1\)).
- **Optimization:** LBFGS (fast quasi-Newton), deterministic with `random_state=42`.  
- **Class imbalance:** `class_weight='balanced'` re-weights classes inversely to frequency.  
- **Decision rule:** Predict **benign** if \(p(y=1\mid x) \ge 0.5\) (threshold can be tuned).

**Why this model?**
- Strong baseline for low-dimensional tabular data.
- Interpretable: after standardization, each coefficient shows effect of a **1-SD** feature increase on log-odds of “benign” (odds ratio = \(e^{\text{coef}}\)).
- Well-calibrated probabilities, easy deployment, and lower overfitting risk on small datasets.

---

## Training & Evaluation Protocol

1. **Split**: Stratified Train/Test (e.g., 80/20).  
2. **Fit**: Pipeline (`StandardScaler → LogisticRegression`) on **train only**.  
3. **Sanity CV**: Optional 5-fold cross-validation on the **train split** to estimate variance (no peeking at test).  
4. **Test metrics** on held-out data:
   - **Accuracy**
   - **ROC AUC** (via `predict_proba`)
   - **Confusion Matrix**
   - **Classification Report** (precision/recall/F1 per class)
   - Optional **ROC curve** plot
5. **Outputs for UI**: Dump held-out test set with predictions to `heldout_test.json`.

---

## Reasoning & Alternatives

- The 30 engineered cytology features are informative; a **linear separator** often performs very well.  
- If malignant recall is paramount, adjust the **probability threshold** (e.g., 0.4) or increase the malignant class weight.  
- **Alternatives to consider**:
  - **Linear SVM** / **RidgeClassifier** (similar bias-variance profile).  
  - **Tree ensembles** (Random Forest, Gradient Boosting) for non-linear patterns (less interpretable, sometimes higher accuracy).  
  - **Calibration** (Platt/Isotonic) if switching to non-probabilistic learners.

---

## Reproducibility & Usage

### Train & export artifacts
```bash
# Install deps
pip install scikit-learn matplotlib joblib numpy

# Train with held-out test split; save model + heldout JSON
python tumor_classifier.py --test-size 0.20 --no-plot
# -> tumor_model.joblib, heldout_test.json
