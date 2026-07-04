import pytest
from fastapi.testclient import TestClient
from src.main import app

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient for testing endpoints."""
    with TestClient(app) as c:
        yield c
