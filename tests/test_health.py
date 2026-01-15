from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    """Test that /health returns 200 with ok: true."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_health_response_format():
    """Test that /health response has correct structure."""
    response = client.get("/health")
    data = response.json()
    assert "ok" in data
    assert isinstance(data["ok"], bool)
