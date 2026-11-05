"""Train the SignBridge classifier from dataset.csv."""
from pathlib import Path
import csv
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset.csv"
MODEL = BASE / "signbridge_model.pkl"


def normalize_row(values):
    points = np.asarray(values, dtype=np.float32).reshape(21, 3)
    points -= points[0]
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale > 1e-6:
        points /= scale
    return points.flatten()


def main():
    if not DATASET.exists():
        raise SystemExit("dataset.csv not found. Collect samples with collect_data.py first.")
    features, labels = [], []
    with DATASET.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        expected = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
        if not reader.fieldnames or not all(c in reader.fieldnames for c in ["label"] + expected):
            raise SystemExit("dataset.csv has an unexpected format. Re-collect data with the updated collect_data.py.")
        for row in reader:
            try:
                label = row["label"].strip().upper()
                values = [float(row[c]) for c in expected]
                if label:
                    labels.append(label)
                    features.append(normalize_row(values))
            except (TypeError, ValueError, KeyError):
                continue

    if len(features) < 10:
        raise SystemExit(f"Only {len(features)} valid rows found. Collect at least 10 samples per sign; 50-100+ is a better starting point.")
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        raise SystemExit("At least two different sign labels are required to train a classifier.")
    print("Samples per label:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")

    X, y = np.asarray(features), np.asarray(labels)
    model = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=1, class_weight="balanced", random_state=42, n_jobs=-1)
    # Use a held-out score only when each class has enough samples for stratification.
    if counts.min() >= 2 and len(X) >= 10:
        test_count = max(int(np.ceil(0.2 * len(X))), len(unique))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_count, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        print(f"Hold-out accuracy: {accuracy_score(y_test, model.predict(X_test)):.1%}")
        print(classification_report(y_test, model.predict(X_test), zero_division=0))
        # Refit on all collected data for deployment.
    model.fit(X, y)
    joblib.dump(model, MODEL)
    print(f"\nModel saved to: {MODEL}")
    print("Supported labels:", ", ".join(model.classes_))


if __name__ == "__main__":
    main()
