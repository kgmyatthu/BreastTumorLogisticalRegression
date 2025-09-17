
from __future__ import annotations
import argparse, json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )


def parse_args():
    p = argparse.ArgumentParser(description="Breast cancer benign/malignant classifier")
    p.add_argument("--test-size", type=float, default=0.20,
                   help="Fraction held out for testing (default: 0.20)")
    p.add_argument("--no-plot", action="store_true", help="Disable ROC plot")
    p.add_argument("--save-test-json", default="heldout_test.json",
                   help="Path to save held-out (non-training) set with predictions (JSON)")
    return p.parse_args()


def main():
    args = parse_args()

    data = load_breast_cancer()
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign
    target_names = list(data.target_names)        # ['malignant', 'benign']
    feature_names = list(data.feature_names)      # 30 names
    n = X.shape[0]
    idx = np.arange(n)                            # keep original row indices

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=args.test_size, stratify=y, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    model = build_model()
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy (train only): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # P(benign)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest Accuracy: {acc:.3f}")
    print(f"Test ROC AUC:  {auc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    sample = X_test[0].reshape(1, -1)
    pred = model.predict(sample)[0]
    pred_proba = model.predict_proba(sample)[0, 1]
    print("\nDemo prediction on one *test* example:")
    print(f"Predicted class: {target_names[pred]} (P[benign]={pred_proba:.3f})")

    try:
        import joblib
        joblib.dump(
            {"model": model, "feature_names": feature_names, "target_names": target_names},
            "tumor_model.joblib",
        )
        print('Saved model to "tumor_model.joblib"')
    except Exception as e:
        print(f"Could not save model (optional): {e}")

    heldout_records = []
    for i in range(X_test.shape[0]):
        feats = {feature_names[j]: float(X_test[i, j]) for j in range(X_test.shape[1])}
        prob_benign = float(y_proba[i])
        rec = {
            "id": int(idx_test[i]),                          # original row index
            "true_label": int(y_test[i]),                    # 0/1
            "true_label_name": target_names[y_test[i]],
            "pred_label": int(y_pred[i]),                    # 0/1
            "pred_label_name": target_names[y_pred[i]],
            "prob_benign": prob_benign,
            "prob_malignant": float(1.0 - prob_benign),
            "features": feats,
        }
        heldout_records.append(rec)

    payload = {
        "meta": {
            "dataset": "Breast Cancer Wisconsin (Diagnostic) – sklearn",
            "n_total": int(n),
            "test_size_fraction": float(args.test_size),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "target_names": target_names,
        },
        "heldout_samples": heldout_records,
    }

    with open(args.save_test_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f'Wrote held-out set with predictions → "{args.save_test_json}"')


if __name__ == "__main__":
    print("Training breast cancer classifier...");
    main()
