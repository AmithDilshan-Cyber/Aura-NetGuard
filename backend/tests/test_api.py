from fastapi.testclient import TestClient

from backend.app.main import app


def test_devices_and_stats_endpoints():
    with TestClient(app) as client:
        r = client.get("/api/devices")
        assert r.status_code == 200
        devices = r.json()
        assert len(devices) > 0
        assert "device_id" in devices[0]

        r2 = client.get("/api/stats")
        assert r2.status_code == 200
        stats = r2.json()
        assert stats["n_devices"] == len(devices)

        r3 = client.get("/api/alerts")
        assert r3.status_code == 200
        assert isinstance(r3.json(), list)


def test_unknown_device_metrics_404():
    with TestClient(app) as client:
        r = client.get("/api/devices/does-not-exist/metrics")
        assert r.status_code == 404


def test_unknown_alert_acknowledge_404():
    with TestClient(app) as client:
        r = client.post("/api/alerts/does-not-exist/acknowledge")
        assert r.status_code == 404


def test_index_serves_dashboard():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
