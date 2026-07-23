import pytest
import os
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.main import app

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient for testing endpoints."""
    with TestClient(app) as c:
        yield c

@pytest.fixture(autouse=True)
def mock_gemini_key_for_offline_tests(request):
    """Dynamically removes GEMINI_API_KEY for all tests except test_gemini_dialogue."""
    if request.module and "test_gemini_dialogue" in request.module.__name__:
        yield
    else:
        with patch.dict(os.environ, {}, clear=False):
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            yield
