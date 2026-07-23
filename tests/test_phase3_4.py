import pytest
import os
from unittest.mock import patch, MagicMock
from src.nlu.extractor import LLMEntityExtractor
from src.dialogue.manager import DialogueManager, DialogueState

@pytest.fixture
def extractor():
    return LLMEntityExtractor()

@pytest.fixture
def dialogue_manager():
    return DialogueManager()

def test_extractor_offline_rules(extractor):
    # Test Name extraction
    res = extractor.extract("Namaste my name is Amit")
    assert res["customer_name"] == "Amit"

    # Test Product extraction
    res = extractor.extract("I ordered a jacket and a shirt")
    assert res["product_name"] in ["jacket", "shirt"]

    # Test Reason extraction
    res = extractor.extract("The laptop received was broken and defective")
    assert res["reason"] in ["broken", "defective"]

    # Test Payment method extraction
    res = extractor.extract("I paid via credit card and upi")
    assert res["payment_method"] in ["credit card", "upi"]

    # Test missing fields
    res = extractor.extract("nothing to extract here")
    assert res["customer_name"] is None
    assert res["order_id"] is None

@patch("httpx.Client.post")
def test_extractor_llm_api_routing(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [{"text": '{"customer_name": "Rohan", "product_name": "jeans", "order_id": "ORD-123456"}'}]
            }
        }]
    }
    mock_post.return_value = mock_resp

    extractor = LLMEntityExtractor()
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
        res = extractor.extract("My name is Rohan and I ordered jeans with order 123456")
        assert res["customer_name"] == "Rohan"
        assert res["product_name"] == "jeans"
        assert res["order_id"] == "ORD-123456"

def test_dialogue_manager_context_memory(dialogue_manager):
    # Turn 1: Tell name
    res1 = dialogue_manager.process_turn("greeting", {"customer_name": "Amit"}, "Hi my name is Amit")
    session = dialogue_manager.sessions["default"]
    assert session["context"]["customer_name"] == "Amit"

    # Turn 2: Start order status workflow
    res2 = dialogue_manager.process_turn("order_status", {}, "I want to track my order")
    # Name should STILL be in context memory!
    assert session["context"]["customer_name"] == "Amit"
    assert session["state"] == DialogueState.AWAITING_ORDER_ID

def test_dialogue_manager_cancel_order_slot_filling(dialogue_manager):
    # Turn 1: Request cancellation without Order ID
    res1 = dialogue_manager.process_turn("cancel_order", {}, "cancel my order please")
    assert dialogue_manager.sessions["default"]["state"] == DialogueState.AWAITING_ORDER_ID
    assert "What is your 6-digit Order ID" in res1["response"]

    # Turn 2: Provide Order ID
    res2 = dialogue_manager.process_turn("unknown", {"order_id": "ORD-876543"}, "876543")
    assert dialogue_manager.sessions["default"]["state"] == DialogueState.IDLE
    assert "successfully cancelled your order ORD-876543" in res2["response"]
    assert res2["tool_executed"] == "cancel_order"

def test_dialogue_manager_refund_slot_filling(dialogue_manager):
    # Turn 1: Request refund
    res1 = dialogue_manager.process_turn("refund_request", {}, "I want a refund")
    assert dialogue_manager.sessions["default"]["state"] == DialogueState.AWAITING_ORDER_ID
    
    # Turn 2: Provide Order ID
    res2 = dialogue_manager.process_turn("unknown", {"order_id": "ORD-876543"}, "876543")
    assert dialogue_manager.sessions["default"]["state"] == DialogueState.AWAITING_REASON
    assert "reason for the refund" in res2["response"]

    # Turn 3: Provide Reason
    res3 = dialogue_manager.process_turn("unknown", {"reason": "damaged"}, "The item was damaged")
    assert dialogue_manager.sessions["default"]["state"] == DialogueState.IDLE
    assert "processed a refund request" in res3["response"]
    assert "damaged" in res3["response"]
    assert res3["tool_executed"] == "refund_order"

def test_dialogue_manager_intent_interruption(dialogue_manager):
    # Turn 1: Start refund request
    res1 = dialogue_manager.process_turn("refund_request", {}, "I want a refund")
    assert dialogue_manager.sessions["default"]["state"] == DialogueState.AWAITING_ORDER_ID

    # Turn 2: Interrupt with password reset
    res2 = dialogue_manager.process_turn("password_reset", {}, "actually reset my password")
    assert dialogue_manager.sessions["default"]["state"] == DialogueState.AWAITING_EMAIL
    assert "registered email address" in res2["response"]
    assert dialogue_manager.sessions["default"]["context"].get("workflow") == "password_reset"
