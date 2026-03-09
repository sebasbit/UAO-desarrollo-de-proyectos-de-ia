from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.domain.categories import CATEGORY_KEYS
from src.inference import model

DATA_DIR = Path("data/raw")
MODELS_DIR = Path("models")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def mlflow_defaults() -> tuple[str, str, str]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "triage-soporte-ti")
    run_name = os.getenv("MLFLOW_RUN_NAME", "")
    return tracking_uri, experiment, run_name


def load_dataset() -> tuple[np.ndarray, list[str]]:
    embeddings: list[np.ndarray] = []
    labels: list[str] = []

    for category_key in CATEGORY_KEYS:
        category_dir = DATA_DIR / category_key
        if not category_dir.is_dir():
            print(f"[aviso] Carpeta no encontrada: {category_dir}")
            continue

        image_paths = [
            path
            for path in category_dir.iterdir()
            if path.suffix.lower() in VALID_EXTENSIONS
        ]
        if not image_paths:
            print(f"[aviso] Sin imágenes en: {category_dir}")
            continue

        print(f"{category_key}: {len(image_paths)} imágenes")
        for image_path in image_paths:
            try:
                image = Image.open(image_path)
                embeddings.append(model.extract_embedding(image))
                labels.append(category_key)
            except Exception as exc:
                print(f"[error] {image_path.name}: {exc}")

    if not embeddings:
        raise RuntimeError("No se encontraron imágenes válidas en data/raw/.")

    return np.array(embeddings), labels


def main() -> None:
    print("=== Pipeline de entrenamiento — Triage TI ===")
    X, y = load_dataset()
    print(f"Total imágenes: {len(y)} | Categorías: {len(set(y))}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
    acc = float(accuracy_score(y_val, y_pred))
    report = classification_report(y_val, y_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=sorted(set(y)))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "encoder": None}, MODELS_DIR / "classifier.pkl")
    (MODELS_DIR / "labels.json").write_text(
        json.dumps({"labels": sorted(set(y))}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (MODELS_DIR / "val_report.txt").write_text(report, encoding="utf-8")
    np.savetxt(MODELS_DIR / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    tracking_uri, experiment, run_name = mlflow_defaults()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    if not run_name:
        run_name = f"logreg_{len(set(y))}c_{len(y_train)}tr_{len(y_val)}val"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("classifier", "LogisticRegression")
        mlflow.log_param("model_id", model.MODEL_ID)
        mlflow.log_param("train_samples", len(y_train))
        mlflow.log_param("val_samples", len(y_val))
        mlflow.log_param("n_classes", len(set(y)))
        mlflow.log_metric("val_macro_f1", macro_f1)
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_artifact(str(MODELS_DIR / "val_report.txt"), artifact_path="reports")
        mlflow.log_artifact(
            str(MODELS_DIR / "confusion_matrix.csv"), artifact_path="reports"
        )
        mlflow.log_artifact(
            str(MODELS_DIR / "labels.json"), artifact_path="model_files"
        )
        mlflow.log_artifact(
            str(MODELS_DIR / "classifier.pkl"), artifact_path="model_files"
        )
        mlflow.sklearn.log_model(clf, artifact_path="sklearn_model")

    print(report)
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("Artefactos guardados en models/")


if __name__ == "__main__":
    main()
