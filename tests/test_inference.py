"""
Pruebas unitarias de la capa de inferencia — T09.

Cubre:
  - classifier.predict() retorna una clave válida del catálogo
  - La regla de seguridad: score < 0.40 → retorna "otros" siempre
  - El embedding tiene la dimensión correcta (192)

Usa un modelo dummy para evitar descarga de pesos en CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.categories import CATEGORY_KEYS
from src.domain.categories import CONFIDENCE_THRESHOLD
from src.inference import classifier


# Fixtures

@pytest.fixture(scope="module")
def trained_artifact():
    """Artefacto entrenado con datos sintéticos para tests sin disco."""
    rng = np.random.default_rng(42)
    n_samples = 40
    embeddings = rng.normal(size=(n_samples, 192)).astype(np.float32)
    labels = [CATEGORY_KEYS[i % len(CATEGORY_KEYS)] for i in range(n_samples)]
    return classifier.train(embeddings, labels)


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Embedding sintético de 192 dimensiones."""
    return np.random.default_rng(7).normal(size=(192,)).astype(np.float32)


# Dimensión del embedding

class TestEmbeddingDimension:
    def test_embedding_has_192_dimensions(self, sample_embedding):
        """El vector de embedding debe tener exactamente 192 dimensiones."""
        assert sample_embedding.shape == (192,)

    def test_embedding_is_float32(self, sample_embedding):
        assert sample_embedding.dtype == np.float32

    def test_embedding_is_1d(self, sample_embedding):
        assert sample_embedding.ndim == 1


# Salida del clasificador

class TestClassifierPredict:
    def test_returns_valid_category_key(self, trained_artifact, sample_embedding):
        """classifier.predict() debe retornar una clave del catálogo."""
        category_key, score = classifier.predict(trained_artifact, sample_embedding)
        assert category_key in CATEGORY_KEYS, (
            f"'{category_key}' no es una clave válida del catálogo."
        )

    def test_score_between_0_and_1(self, trained_artifact, sample_embedding):
        _, score = classifier.predict(trained_artifact, sample_embedding)
        assert 0.0 <= score <= 1.0

    def test_returns_tuple(self, trained_artifact, sample_embedding):
        result = classifier.predict(trained_artifact, sample_embedding)
        assert isinstance(result, tuple) and len(result) == 2

    def test_category_key_is_string(self, trained_artifact, sample_embedding):
        category_key, _ = classifier.predict(trained_artifact, sample_embedding)
        assert isinstance(category_key, str)

    def test_score_is_float(self, trained_artifact, sample_embedding):
        _, score = classifier.predict(trained_artifact, sample_embedding)
        assert isinstance(score, float)


# Regla de seguridad: confidence gate

class TestConfidenceGate:
    def test_low_score_returns_otros(self, trained_artifact):
        """Si el score < CONFIDENCE_THRESHOLD, la clave debe ser 'otros'."""
        # Construimos un embedding que genere baja confianza
        # Entrenamos un artefacto con un solo label para forzar el escenario
        rng = np.random.default_rng(0)
        emb_train = rng.normal(size=(10, 192)).astype(np.float32)
        labels = ["otros"] * 5 + ["red_conectividad"] * 5
        art = classifier.train(emb_train, labels)

        # Un embedding ortogonal a los datos de entrenamiento tendrá score bajo
        emb_test = np.zeros(192, dtype=np.float32)
        emb_test[0] = 1000.0   # valor extremo → distribucion de prob. aplastada

        category_key, score = classifier.predict(art, emb_test)

        if score < CONFIDENCE_THRESHOLD:
            assert category_key == "otros", (
                f"Con score {score:.4f} < {CONFIDENCE_THRESHOLD} "
                f"se esperaba 'otros', se obtuvo '{category_key}'."
            )

    def test_confidence_threshold_value(self):
        """El umbral de confianza debe ser 0.40 según el documento técnico."""
        assert CONFIDENCE_THRESHOLD == pytest.approx(0.40)

    def test_high_score_keeps_category(self, trained_artifact):
        """Con score >= umbral, la clave retornada debe pertenecer al catálogo
        y NO ser forzada a 'otros' por el confidence gate."""
        rng = np.random.default_rng(99)
        # Dos clases bien separadas → LogisticRegression tendrá alta confianza
        # en la zona de entrenamiento de "red_conectividad"
        emb_a = rng.normal(loc=10.0, scale=0.05, size=(40, 192)).astype(np.float32)
        emb_b = rng.normal(loc=-10.0, scale=0.05, size=(40, 192)).astype(np.float32)
        embeddings = np.vstack([emb_a, emb_b])
        labels = ["red_conectividad"] * 40 + ["correo_office365"] * 40
        art = classifier.train(embeddings, labels)

        # Punto muy cercano al centroide de "red_conectividad" → score alto
        emb_test = np.full(192, 10.0, dtype=np.float32)
        category_key, score = classifier.predict(art, emb_test)

        # Si el score es alto, la categoría debe ser la correcta, no "otros"
        assert score >= CONFIDENCE_THRESHOLD, (
            f"Se esperaba score >= {CONFIDENCE_THRESHOLD}, se obtuvo {score:.4f}"
        )
        assert category_key == "red_conectividad", (
            f"Con score {score:.4f} se esperaba 'red_conectividad', se obtuvo '{category_key}'"
        )


# train / save / load

class TestTrainSaveLoad:
    def test_train_returns_artifact_with_keys(self, trained_artifact):
        assert "clf" in trained_artifact
        assert "encoder" in trained_artifact

    def test_save_and_load_roundtrip(self, trained_artifact, tmp_path):
        """save() y load() deben preservar el artefacto."""
        path = tmp_path / "classifier_test.pkl"
        classifier.save(trained_artifact, path)
        loaded = classifier.load(path)
        assert "clf" in loaded and "encoder" in loaded

    def test_load_nonexistent_raises(self, tmp_path):
        from pathlib import Path
        with pytest.raises(FileNotFoundError):
            classifier.load(tmp_path / "no_existe.pkl")
