from fastapi.testclient import TestClient


def test_predict_rejects_non_image(client: TestClient) -> None:
    files = {"image": ("test.txt", b"hola", "text/plain")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 400


def test_predict_returns_valid_payload(
    client: TestClient,
    sample_image_bytes: bytes,
) -> None:
    files = {"image": ("sample.png", sample_image_bytes, "image/png")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) >= 1
    assert "category" in data["predictions"][0]
    assert "category_key" in data["predictions"][0]
    assert "team" in data["predictions"][0]
    assert "timestamp" in data
