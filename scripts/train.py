"""
Pipeline de entrenamiento del clasificador DeiT-Tiny + sklearn.

Flujo:
  1. Lee el dataset desde data/raw/ (una carpeta por categoría)
  2. Extrae embeddings con DeiT-Tiny para cada imagen
  3. Entrena LogisticRegression sobre los embeddings
  4. Guarda el artefacto en models/classifier.pkl
  5. Imprime métricas sobre el split de validación

Estructura esperada del dataset:
  data/raw/
  ├── red_conectividad/
  │   ├── img_001.png
  │   └── img_002.jpg
  ├── acceso_contrasenas/
  │   └── ...
  └── (una carpeta por cada clave en CATEGORY_KEYS)

Uso:
  make train
  uv run scripts/train.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Añade src/ al path para poder importar los módulos del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.domain.categories import CATEGORY_KEYS
from src.inference import classifier
from src.inference import model

DATA_DIR = Path("data/raw")
MODEL_PATH = Path("models/classifier.pkl")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_dataset() -> tuple[list[np.ndarray], list[str]]:
    """
    Recorre data/raw/ y extrae embeddings + etiquetas para todas las imágenes.

    Returns:
        embeddings : lista de vectores numpy (192,)
        labels     : lista de claves de categoría correspondientes
    """
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    total = 0

    for category_key in CATEGORY_KEYS:
        category_dir = DATA_DIR / category_key
        if not category_dir.is_dir():
            print(f"  [aviso] Carpeta no encontrada: {category_dir}")
            continue

        image_paths = [
            p for p in category_dir.iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS
        ]

        if not image_paths:
            print(f"  [aviso] Sin imágenes en: {category_dir}")
            continue

        print(f"  {category_key}: {len(image_paths)} imágenes")

        for img_path in image_paths:
            try:
                image = Image.open(img_path)
                embedding = model.extract_embedding(image)
                embeddings.append(embedding)
                labels.append(category_key)
                total += 1
            except Exception as e:
                print(f"    [error] {img_path.name}: {e}")

    return embeddings, labels


def main() -> None:
    print("=" * 50)
    print("Pipeline de entrenamiento — Triage TI")
    print("=" * 50)

    # 1. Cargar dataset
    print("\n[1/4] Cargando dataset desde data/raw/ ...")
    embeddings_list, labels = load_dataset()

    if not embeddings_list:
        print("\n[ERROR] No se encontraron imágenes. Verifica la estructura del dataset.")  # noqa: E501
        sys.exit(1)

    X = np.array(embeddings_list)
    y = labels
    print(f"  Total: {len(y)} imágenes | {len(set(y))} categorías")

    # 2. Split train/val (85% / 15% estratificado)
    print("\n[2/4] Dividiendo en train / validación (85% / 15%) ...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"  Train: {len(y_train)} | Val: {len(y_val)}")

    # 3. Entrenar clasificador
    print("\n[3/4] Entrenando LogisticRegression ...")
    artifact = classifier.train(X_train, y_train)
    print("  Entrenamiento completado.")

    # 4. Evaluar en validación
    print("\n[4/4] Evaluando en validación ...")
    clf = artifact["clf"]
    encoder = artifact["encoder"]
    y_pred_encoded = clf.predict(X_val)
    y_pred = encoder.inverse_transform(y_pred_encoded).tolist()

    print("\n" + classification_report(y_val, y_pred, zero_division=0))

    # 5. Guardar artefacto
    classifier.save(artifact, MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()
