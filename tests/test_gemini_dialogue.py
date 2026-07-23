import pytest
import os
import json
from unittest.mock import patch, MagicMock
from src.dialogue.manager import DialogueManager, DialogueState
from src.database.db_manager import init_db

@pytest.fixture(autouse=True)
def setup_db():
    init_db()

def test_dialogue_manager_fallback_turn_when_no_key():
    # If GEMINI_API_KEY is not in environment, it should use the process_fallback_turn and match backward behavior
    with patch.dict(os.environ, {}, clear=True):
        dm = DialogueManager()
        res = dm.process_turn("order_status", {}, "I want to track my order")
        assert dm.sessions["default"]["state"] == DialogueState.AWAITING_ORDER_ID
        assert "Order ID" in res["response"]

@patch("httpx.Client.post")
def test_dialogue_manager_gemini_slot_prompting(mock_post):
    # Mock first turn where user asks for order status, but Order ID is missing
    mock_resp1 = MagicMock()
    mock_resp1.status_code = 200
    mock_resp1.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "intent": "order_status",
                        "tool": None,
                        "entities": {
                            "order_id": None
                        },
                        "missing_slots": ["order_id"],
                        "assistant_reply": "Sure, please provide your 6-digit Order ID."
                    })
                }]
            }
        }]
    }
    mock_post.return_value = mock_resp1

    # Simulate turn with GEMINI_API_KEY set
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_gemini_key"}):
        dm = DialogueManager()
        res = dm.process_turn("order_status", {}, "Track order please")
        
        # Verify correct payload structure sent to Gemini API
        assert mock_post.called
        assert res["state"] == DialogueState.AWAITING_ORDER_ID
        assert "Order ID" in res["response"]
        assert res["tool_executed"] is None

@patch("httpx.Client.post")
def test_dialogue_manager_gemini_tool_execution(mock_post):
    # Mock two turns: 
    # 1st call returns JSON payload indicating tool call get_order_status
    # 2nd call (synthesize response) returns natural response string
    mock_resp_json = MagicMock()
    mock_resp_json.status_code = 200
    mock_resp_json.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "intent": "order_status",
                        "tool": "get_order_status",
                        "entities": {
                            "order_id": "ORD-876543"
                        },
                        "missing_slots": [],
                        "assistant_reply": "Let me look up your order status."
                    })
                }]
            }
        }]
    }
    
    mock_resp_text = MagicMock()
    mock_resp_text.status_code = 200
    mock_resp_text.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [{"text": "Your order ORD-876543 has been shipped and is expected on 2026-07-06."}]
            }
        }]
    }

    # Setup mock post to return json mock on first call, and text mock on second call
    mock_post.side_effect = [mock_resp_json, mock_resp_text]

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_gemini_key"}):
        dm = DialogueManager()
        
        # Seed order_id in history
        dm.sessions["default"] = {
            "state": DialogueState.AWAITING_ORDER_ID,
            "context": {"order_id": "ORD-876543"},
            "history": [{"role": "user", "text": "ORD-876543"}]
        }
        
        res = dm.process_turn("unknown", {"order_id": "ORD-876543"}, "ORD-876543")
        
        # Assertions
        assert mock_post.call_count == 2
        assert res["state"] == DialogueState.IDLE
        assert "shipped" in res["response"]
        assert res["tool_executed"] == "get_order_status"
        assert res["tool_result"]["success"] is True
        assert res["tool_result"]["status"] == "Shipped"
