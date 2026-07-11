import pytest
from src.nlu.classifier import MockIntentClassifier
from src.nlu.extractor import MockEntityExtractor
from src.dialogue.manager import DialogueManager, DialogueState

def test_intent_classifier():
    classifier = MockIntentClassifier()
    
    # Test Greeting
    res = classifier.classify("Hello, how are you?")
    assert res["intent"] == "greeting"
    assert res["confidence"] >= 0.8
    
    # Test Order Status
    res = classifier.classify("where is my order?")
    assert res["intent"] == "order_status"
    assert res["confidence"] >= 0.8
    
    # Test Password Reset
    res = classifier.classify("I want to reset my password")
    assert res["intent"] == "password_reset"
    
    # Test Update Address
    res = classifier.classify("update my shipping address please")
    assert res["intent"] == "update_address"
    
    # Test Farewell
    res = classifier.classify("Thank you, bye bye")
    assert res["intent"] == "farewell"
    
    # Test Unknown
    res = classifier.classify("what is the weather today?")
    assert res["intent"] == "unknown"
    assert res["confidence"] == 0.0

def test_entity_extractor():
    extractor = MockEntityExtractor()
    
    # Extract Order ID
    res = extractor.extract("My order ID is 123456")
    assert res.get("order_id") == "ORD-123456"
    
    # Extract Phone
    res = extractor.extract("Call me on 9876543210 please")
    assert res.get("phone_number") == "9876543210"
    
    # Extract Email
    res = extractor.extract("My email is alex@gmail.com")
    assert res.get("email") == "alex@gmail.com"
    
    # Extract Address
    res = extractor.extract("Please update my address to 123 Main Street")
    assert res.get("address") == "123 Main Street"

def test_dialogue_manager_flow():
    manager = DialogueManager()
    session_id = "test_sess_1"
    
    # Turn 1: Initial greeting
    res = manager.process_turn("greeting", {}, "Hi", session_id=session_id)
    assert manager.sessions[session_id]["state"] == DialogueState.IDLE
    assert "Welcome to Vani" in res["response"]
    
    # Turn 2: Query order status without ID
    res = manager.process_turn("order_status", {}, "I want to check my order", session_id=session_id)
    assert manager.sessions[session_id]["state"] == DialogueState.AWAITING_ORDER_ID
    assert "order status" in res["response"]
    
    # Turn 3: Provide Order ID
    res = manager.process_turn("unknown", {"order_id": "ORD-876543"}, "My order number is 876543", session_id=session_id)
    assert manager.sessions[session_id]["state"] == DialogueState.IDLE
    assert res["tool_executed"] == "get_order_status"
    assert res["tool_result"]["success"] is True
    assert "Shipped" in res["response"]
