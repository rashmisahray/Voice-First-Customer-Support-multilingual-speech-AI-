import pytest
import os
from unittest.mock import patch, MagicMock
from src.nlu.classifier import MockIntentClassifier
from src.nlu.llm_fallback import LLMFallback

def test_new_intents_classification():
    classifier = MockIntentClassifier()

    # Test cancel_order
    res = classifier.classify("Please cancel my order ORD-123456")
    assert res["intent"] == "cancel_order"
    assert res["confidence"] >= 0.7

    # Test human_agent
    res = classifier.classify("I want to speak with a live representative")
    assert res["intent"] == "human_agent"
    assert res["confidence"] >= 0.7

    # Test payment_issue
    res = classifier.classify("My transaction failed and payment declined")
    assert res["intent"] == "payment_issue"
    assert res["confidence"] >= 0.7

    # Test store_hours
    res = classifier.classify("What are the store timings and open hours?")
    assert res["intent"] == "store_hours"
    assert res["confidence"] >= 0.7

def test_local_fallback_responses():
    fallback = LLMFallback()

    # Test store location mapping
    res = fallback.generate_response("Where is the store located?")
    assert "124 MG Road" in res

    # Test return policy mapping
    res = fallback.generate_response("What is your return policy?")
    assert "30 days" in res

    # Test cancel order mapping
    res = fallback.generate_response("cancel my order please")
    assert "cancel" in res.lower()
    assert "order id" in res.lower()

    # Test general unknown query fallback
    res = fallback.generate_response("What is the meaning of life?")
    assert "order status" in res
    assert "reset password" in res

@patch("httpx.Client.post")
def test_gemini_api_call(mock_post):
    # Mock Gemini response structure
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [{"text": "Gemini: Hello user!"}]
            }
        }]
    }
    mock_post.return_value = mock_resp

    fallback = LLMFallback()
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_gemini_key"}):
        res = fallback.generate_response("Namaste Gemini")
        assert res == "Gemini: Hello user!"
        
        # Verify it targeted the right URL
        args, kwargs = mock_post.call_args
        assert "generativelanguage.googleapis.com" in args[0]
        assert "fake_gemini_key" in args[0]

@patch("httpx.Client.post")
def test_openai_api_call(mock_post):
    # Mock OpenAI response structure
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": "OpenAI: Hello user!"
            }
        }]
    }
    mock_post.return_value = mock_resp

    fallback = LLMFallback()
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake_openai_key"}):
        res = fallback.generate_response("Hi GPT")
        assert res == "OpenAI: Hello user!"
        
        # Verify it targeted the right URL
        args, kwargs = mock_post.call_args
        assert "api.openai.com" in args[0]
        headers = kwargs.get("headers", {})
        assert "Bearer fake_openai_key" in headers.get("Authorization", "")
