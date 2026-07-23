import pytest
from src.dialogue.manager import DialogueManager, DialogueState
from src.nlu.classifier import MockIntentClassifier
from src.tools.backend_client import BackendClient
from src.database.db_manager import init_db

@pytest.fixture(autouse=True)
def init_test_db():
    init_db()

def test_order_status_slot_filling():
    dm = DialogueManager()
    
    # 1. Trigger order status without ID
    res1 = dm.process_turn("order_status", {}, "I want to check my order status")
    assert dm.sessions["default"]["state"] == DialogueState.AWAITING_ORDER_ID
    assert "Order ID" in res1["response"]

    # 2. Provide Order ID (clean number)
    res2 = dm.process_turn("unknown", {"order_id": "ORD-876543"}, "876543")
    assert dm.sessions["default"]["state"] == DialogueState.IDLE
    assert "Shipped" in res2["response"]
    assert res2["tool_executed"] == "get_order_status"

def test_password_reset_slot_filling():
    dm = DialogueManager()

    # 1. Trigger password reset
    res1 = dm.process_turn("password_reset", {}, "Mera password reset kar do")
    assert dm.sessions["default"]["state"] == DialogueState.AWAITING_EMAIL
    assert "registered email address" in res1["response"]

    # 2. Provide email
    res2 = dm.process_turn("unknown", {"email": "john@example.com"}, "john@example.com")
    assert dm.sessions["default"]["state"] == DialogueState.IDLE
    assert "password reset" in res2["response"].lower()
    assert res2["tool_executed"] == "reset_password"

def test_update_address_slot_filling():
    dm = DialogueManager()

    # 1. Trigger update address
    res1 = dm.process_turn("update_address", {}, "Address badalna hai")
    assert dm.sessions["default"]["state"] == DialogueState.AWAITING_PHONE
    assert "registered phone number" in res1["response"]

    # 2. Provide Phone
    res2 = dm.process_turn("unknown", {"phone_number": "9876543210"}, "9876543210")
    assert dm.sessions["default"]["state"] == DialogueState.AWAITING_ADDRESS
    assert "delivery address" in res2["response"]

    # 3. Provide Address
    res3 = dm.process_turn("unknown", {"address": "789 Sea Face Rd, Mumbai"}, "789 Sea Face Rd, Mumbai")
    assert dm.sessions["default"]["state"] == DialogueState.IDLE
    assert "successfully updated" in res3["response"].lower()
    assert res3["tool_executed"] == "update_address"

def test_active_intent_lock_during_slot_filling():
    dm = DialogueManager()
    classifier = MockIntentClassifier()

    # 1. Start order status workflow
    dm.process_turn("order_status", {}, "Track my order")
    assert dm.sessions["default"]["state"] == DialogueState.AWAITING_ORDER_ID
    assert dm.sessions["default"]["context"].get("workflow") == "order_status"

    # 2. Send slot value "876543" which previously might trigger keyword collision.
    # Verify keyword classifier does not match "876543" as password_reset or update_address!
    nlu_res = classifier.classify("876543")
    assert nlu_res["intent"] == "unknown"

    # 3. Process turn with classified "unknown" intent.
    # Active intent ("order_status") should be locked and NOT switch to password_reset.
    res = dm.process_turn(nlu_res["intent"], {"order_id": "ORD-876543"}, "876543")
    assert dm.sessions["default"]["state"] == DialogueState.IDLE
    assert res["tool_executed"] == "get_order_status"
    assert "Shipped" in res["response"]

def test_explicit_interruption_during_slot_filling():
    dm = DialogueManager()

    # 1. Start order status
    dm.process_turn("order_status", {}, "Track order")
    assert dm.sessions["default"]["state"] == DialogueState.AWAITING_ORDER_ID

    # 2. Explicitly interrupt with password reset
    res = dm.process_turn("password_reset", {}, "actually reset my password please")
    assert dm.sessions["default"]["state"] == DialogueState.AWAITING_EMAIL
    assert "registered email address" in res["response"]
    assert dm.sessions["default"]["context"].get("workflow") == "password_reset"
