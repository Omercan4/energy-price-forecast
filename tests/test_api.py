import json

from fastapi.testclient import TestClient

from epf import config
from epf.serve import app as app_module


def client_with_docs(tmp_path, monkeypatch, files):
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
    monkeypatch.setattr(config, "DOCS_DIR", tmp_path)
    return TestClient(app_module.app)


def test_forecast_endpoint_returns_the_latest_file(tmp_path, monkeypatch):
    client = client_with_docs(tmp_path, monkeypatch, {"forecast.json": {"target_day": "2026-10-25", "hours": []}})

    response = client.get("/forecast")

    assert response.status_code == 200 and response.json()["target_day"] == "2026-10-25"


def test_missing_files_give_a_helpful_404(tmp_path, monkeypatch):
    client = client_with_docs(tmp_path, monkeypatch, {})

    response = client.get("/health")

    assert response.status_code == 404 and "does not exist yet" in response.json()["detail"]


def test_an_old_forecast_can_be_read_by_date(tmp_path, monkeypatch):
    client = client_with_docs(tmp_path, monkeypatch, {"forecasts/2026-10-24.json": {"target_day": "2026-10-24"}})

    assert client.get("/forecast/2026-10-24").json()["target_day"] == "2026-10-24"


def test_path_tricks_in_the_date_are_rejected(tmp_path, monkeypatch):
    client = client_with_docs(tmp_path, monkeypatch, {"forecast.json": {}})

    assert client.get("/forecast/..%2F..%2Fsecret").status_code in (400, 404)
    assert client.get("/forecast/not-a-date").status_code == 400
