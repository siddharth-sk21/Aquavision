"""
AquaVision - ML Training Script
================================
Generates synthetic training data from hydrogeological rules
and trains a RandomForest classifier.

Run this ONCE before starting the app:
    python train_model.py

Output:
    aqua_model.pkl  ← loaded by classifier.py at runtime
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from rules import score_inputs

# ─── Synthetic Data Generator ──────────────────────────────────────────────────

def generate_synthetic_data(n: int = 5000) -> pd.DataFrame:
    """
    Generates n synthetic data points based on hydrogeological rules.
    This is our 'Base Knowledge' — mimics the rule-based classifier.

    Why synthetic?
    → No labeled borewell dataset exists publicly for India
    → Our expert rules ARE the domain knowledge
    → RF learns the decision boundary more smoothly than if/else
    """
    np.random.seed(42)  # reproducible results
    data = []

    for _ in range(n):
        # Random realistic Indian agricultural land values
        ndvi      = np.clip(np.random.normal(0.3, 0.15), -1, 1)
        ndwi      = np.clip(np.random.normal(0.0, 0.2), -1, 1)
        ndmi      = np.clip(np.random.normal(0.0, 0.15), -1, 1)
        elevation = np.clip(np.random.normal(400, 250), 0, np.inf)
        slope     = np.clip(np.random.normal(10, 6), 0, np.inf)

        # Apply the same weighted scoring used by classifier.py.
        _, _, _, verdict = score_inputs(ndvi, ndwi, ndmi, elevation, slope)

        # 0=LOW, 1=MEDIUM, 2=HIGH
        label = 2 if verdict == "HIGH" else (1 if verdict == "MEDIUM" else 0)

        data.append([ndvi, ndwi, ndmi, elevation, slope, label])

    return pd.DataFrame(
        data,
        columns=["ndvi", "ndwi", "ndmi", "elevation", "slope", "label"]
    )

# ─── Train Model ───────────────────────────────────────────────────────────────

def train_and_save(output_path: str = "aqua_model.pkl"):
    print("🌱 Generating synthetic training data...")
    df = generate_synthetic_data(n=5000)

    X = df.drop("label", axis=1)
    y = df["label"]

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🤖 Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,   # 100 decision trees
        max_depth=8,        # prevents overfitting
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    print("\n📊 Model Performance:")
    print(classification_report(
        y_test, y_pred,
        target_names=["LOW", "MEDIUM", "HIGH"]
    ))

    # ── Feature Importance (great for judges!) ─────────────────────────────────
    features    = ["ndvi", "ndwi", "ndmi", "elevation", "slope"]
    importances = model.feature_importances_
    print("🔬 Feature Importance:")
    for f, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {f:<12} {bar} {imp:.3f}")

    # ── Save model ─────────────────────────────────────────────────────────────
    joblib.dump(model, output_path)
    print(f"\n✅ Model saved to: {output_path}")
    print("👉 Now run app.py to start AquaVision — classifier.py will load this automatically.")


if __name__ == "__main__":
    train_and_save()