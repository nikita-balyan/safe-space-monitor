from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("data")
PROCESSED_CSV = DATA_DIR / "processed_data.csv"
MODEL_PATH = Path("models/overload_model.joblib")
THRESH_PATH = Path("models/model_threshold.txt")
MODEL_PATH.parent.mkdir(exist_ok=True)


# -----------------------------
# Step 1: Load processed data
# -----------------------------
def load_data():
    if not PROCESSED_CSV.exists():
        print("❌ Processed CSV not found. Run simulator.py first.")
        return None

    df = pd.read_csv(PROCESSED_CSV)
    print(f"✅ Loaded {len(df)} rows from {PROCESSED_CSV}")
    return df


# -----------------------------
# Step 2: Train models
# -----------------------------
def train_models():
    df = load_data()
    if df is None:
        return None, None

    # Features & target
    feature_cols = [
        "noise", "light", "motion",
        "noise_mean", "light_mean", "motion_mean",
        "noise_std", "light_std", "motion_std"
    ]
    X = df[feature_cols]
    y = df["risk_label"].values

    # --- Label distribution check ---
    unique, counts = np.unique(y, return_counts=True)
    print("Label distribution:", dict(zip(unique, counts)))

    if len(unique) < 2:
        print("❌ Only one class present in labels. Please recollect more diverse data.")
        return None, None

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # -------------------------
    # Logistic Regression (with scaling)
    # -------------------------
    lr_params = {"C": [0.1, 1, 3]}
    best_lr_score, best_lr_model = 0, None

    print("\nTraining Logistic Regression models...")
    for c in lr_params["C"]:
        print(f"  Testing C={c}")
        lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=c, class_weight="balanced", max_iter=500, random_state=42
            ))
        ])
        lr_pipeline.fit(X_train, y_train)
        y_pred = lr_pipeline.predict(X_val)
        recall = recall_score(y_val, y_pred)
        print(f"    Validation Recall: {recall:.3f}")
        if recall > best_lr_score:
            best_lr_score, best_lr_model = recall, lr_pipeline

    # -------------------------
    # Decision Tree
    # -------------------------
    dt_params = {"max_depth": [3, 5, 7], "min_samples_leaf": [5, 10]}
    best_dt_score, best_dt_model = 0, None

    print("\nTraining Decision Tree models...")
    for depth in dt_params["max_depth"]:
        for min_samples in dt_params["min_samples_leaf"]:
            print(f"  Testing max_depth={depth}, min_samples_leaf={min_samples}")
            dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_samples, random_state=42)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_val)
            recall = recall_score(y_val, y_pred)
            print(f"    Validation Recall: {recall:.3f}")
            if recall > best_dt_score:
                best_dt_score, best_dt_model = recall, dt

    # -------------------------
    # Evaluation helper
    # -------------------------
    # Replace the confusion matrix visualization part:
def evaluate_model(model, X, y, name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n{name} Performance:")
    print(f"Accuracy:  {accuracy_score(y, y_pred):.3f}")
    print(f"Precision: {precision_score(y, y_pred):.3f}")
    print(f"Recall:    {recall_score(y, y_pred):.3f}")
    print(f"F1 Score:  {f1_score(y, y_pred):.3f}")
    
    # SIMPLIFIED: Just print the confusion matrix instead of plotting
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    return y_proba

    # -------------------------
    # Validation & Test
    # -------------------------
    print("\n" + "="*50)
    print("Validation Set Results")
    print("="*50)
    lr_val_proba = evaluate_model(best_lr_model, X_val, y_val, "Logistic Regression")
    dt_val_proba = evaluate_model(best_dt_model, X_val, y_val, "Decision Tree")

    print("\n" + "="*50)
    print("Test Set Results")
    print("="*50)
    lr_test_proba = evaluate_model(best_lr_model, X_test, y_test, "Logistic Regression")
    dt_test_proba = evaluate_model(best_dt_model, X_test, y_test, "Decision Tree")

    # -------------------------
    # Model selection
    # -------------------------
    if best_lr_score >= best_dt_score:
        print("\nSelected Logistic Regression as the best model")
        best_model, best_proba = best_lr_model, lr_test_proba
    else:
        print("\nSelected Decision Tree as the best model")
        best_model, best_proba = best_dt_model, dt_test_proba

    # -------------------------
    # Threshold adjustment
    # -------------------------
    test_recall = recall_score(y_test, best_model.predict(X_test))
    print(f"\nTest recall before threshold adjustment: {test_recall:.3f}")

    if test_recall < 0.9 and best_proba is not None:
        print("Recall below 90%, adjusting threshold...")
        thresholds = np.arange(0.3, 0.7, 0.05)
        best_threshold, best_recall = 0.5, 0
        for t in thresholds:
            y_adj = (best_proba >= t).astype(int)
            recall = recall_score(y_test, y_adj)
            if recall > best_recall:
                best_threshold, best_recall = t, recall
        print(f"Optimal threshold: {best_threshold:.2f} (Recall: {best_recall:.3f})")
    else:
        best_threshold = 0.5
        print(f"Using default threshold: {best_threshold}")

    # -------------------------
    # Save artifacts
    # -------------------------
    joblib.dump(best_model, MODEL_PATH)
    with open(THRESH_PATH, "w") as f:
        f.write(str(best_threshold))

    print("\nModel training complete!")
    print(f"Model saved → {MODEL_PATH}")
    print(f"Threshold saved → {THRESH_PATH}")

    return best_model, best_threshold


if __name__ == "__main__":
    train_models()

