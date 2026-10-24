"""Train and evaluate baseline classification models."""
from __future__ import annotations
import json
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from preprocess import TARGET, clean_dataset, validate_dataset

RANDOM_STATE = 42

def make_pipeline(X: pd.DataFrame, model):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", num_pipe, numeric), ("categorical", cat_pipe, categorical)])
    return Pipeline([("preprocessor", prep), ("model", model)])

def train_and_evaluate(df: pd.DataFrame, output_dir: str = "models"):
    errors = validate_dataset(df)
    if errors:
        raise ValueError(" ".join(errors))
    data = clean_dataset(df)
    if data[TARGET].nunique() < 2:
        raise ValueError("Training needs both Attrition Yes and No records.")
    y = data[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    X = data.drop(columns=[TARGET])
    # Avoid stratification errors on tiny uploaded datasets.
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                                random_state=RANDOM_STATE, n_jobs=-1),
    }
    results, fitted = [], {}
    for name, estimator in candidates.items():
        pipe = make_pipeline(X_train, estimator)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        row = {
            "Model": name,
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "F1-score": float(f1_score(y_test, pred, zero_division=0)),
            "Confusion Matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
            "Classification Report": classification_report(y_test, pred, labels=[0, 1],
                                                            target_names=["No Attrition", "Attrition"],
                                                            output_dict=True, zero_division=0),
        }
        results.append(row)
        fitted[name] = pipe
    # Select using F1 as a balanced metric for the minority Attrition class.
    best = max(results, key=lambda r: (r["F1-score"], r["Recall"], r["Accuracy"]))
    os.makedirs(output_dir, exist_ok=True)
    bundle = {"pipeline": fitted[best["Model"]], "features": X.columns.tolist(),
              "target": TARGET, "best_model": best["Model"], "metrics": best}
    joblib.dump(bundle, os.path.join(output_dir, "attrition_model.joblib"))
    with open(os.path.join(output_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump({"results": results, "best_model": best["Model"], "train_rows": len(X_train),
                   "test_rows": len(X_test), "features": X.columns.tolist()}, f, indent=2)
    return results, bundle, len(X_train), len(X_test)

if __name__ == "__main__":
    source = "data/employee_attrition.csv"
    df = pd.read_csv(source)
    results, bundle, train_n, test_n = train_and_evaluate(df)
    print(f"Training rows: {train_n}; testing rows: {test_n}")
    print(pd.DataFrame([{k: r[k] for k in ["Model", "Accuracy", "Precision", "Recall", "F1-score"]} for r in results]))
    print("Saved model: models/attrition_model.joblib")
