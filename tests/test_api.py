from fastapi.testclient import TestClient

from src import api

client = TestClient(api.app)


class FakeModel:
    classes_ = (0, 1)

    def predict(self, features):
        return [1]

    def predict_proba(self, features):
        return [[0.2, 0.8]]


SAMPLE_FEATURES = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


def test_predict_returns_prediction_and_probabilities(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: FakeModel())

    response = client.post("/predict", json=SAMPLE_FEATURES)

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "confidence": 0.8,
        "disease_probability": 0.8,
    }


def test_predict_rejects_missing_or_unexpected_fields(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: FakeModel())
    missing_age = {key: value for key, value in SAMPLE_FEATURES.items() if key != "age"}
    unexpected_field = {**SAMPLE_FEATURES, "patient_name": "not accepted"}

    assert client.post("/predict", json=missing_age).status_code == 422
    assert client.post("/predict", json=unexpected_field).status_code == 422


def test_predict_accepts_missing_categorical_measurements(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: FakeModel())
    payload = {**SAMPLE_FEATURES, "ca": None, "thal": None}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200


def test_predict_returns_service_unavailable_without_model(monkeypatch):
    def unavailable_model():
        raise RuntimeError("Model is not trained")

    monkeypatch.setattr(api, "load_model", unavailable_model)

    response = client.post("/predict", json=SAMPLE_FEATURES)

    assert response.status_code == 503


def test_health_and_metrics_endpoints_respond():
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200