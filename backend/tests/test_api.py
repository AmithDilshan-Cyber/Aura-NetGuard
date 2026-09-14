from fastapi.testclient import TestClient

from backend.app.main import app


def test_devices_endpoint_lists_the_fleet():
    with TestClient(app) as client:
        devices = client.get("/api/devices").json()
        assert len(devices) > 0
        assert {"device_id", "name", "device_type", "site"} <= devices[0].keys()


def test_stats_endpoint_agrees_with_the_device_list():
    with TestClient(app) as client:
        devices = client.get("/api/devices").json()
        stats = client.get("/api/stats").json()
        assert stats["n_devices"] == len(devices)
        assert stats["active_alerts"] >= 0
        assert "feedback_stats" in stats


def test_alerts_endpoint_returns_a_list():
    with TestClient(app) as client:
        assert isinstance(client.get("/api/alerts").json(), list)


def test_unknown_device_metrics_returns_404():
    with TestClient(app) as client:
        assert client.get("/api/devices/does-not-exist/metrics").status_code == 404


def test_unknown_alert_actions_return_404():
    with TestClient(app) as client:
        assert client.post("/api/alerts/nope/acknowledge").status_code == 404
        assert (
            client.post("/api/alerts/nope/feedback", json={"is_true_positive": True}).status_code
            == 404
        )


def test_feedback_requires_a_valid_body():
    with TestClient(app) as client:
        assert client.post("/api/alerts/nope/feedback", json={}).status_code == 422


def test_index_serves_the_dashboard():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
