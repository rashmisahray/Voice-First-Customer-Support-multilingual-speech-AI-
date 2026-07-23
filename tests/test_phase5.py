import pytest
import os
import sqlite3
from src.database.db_manager import init_db, get_db_connection, DB_PATH
from src.tools.backend_client import BackendClient
from src.dialogue.manager import DialogueManager, DialogueState

@pytest.fixture(autouse=True)
def setup_test_db():
    # Make sure we initialize and clean the database for each test
    init_db()
    yield
    # Clean up test DB after testing
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except Exception:
            pass

def test_database_initialization():
    # Verify that the DB file exists
    assert os.path.exists(DB_PATH)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check customers table
    cursor.execute("SELECT COUNT(*) FROM customers")
    assert cursor.fetchone()[0] == 2

    # Check orders table
    cursor.execute("SELECT COUNT(*) FROM orders")
    assert cursor.fetchone()[0] == 2
    
    conn.close()

def test_backend_client_sqlite_operations():
    client = BackendClient()

    # Test get_order_status
    res = client.get_order_status("ORD-876543")
    assert res["success"] is True
    assert res["status"] == "Shipped"

    # Test update_address updates the database
    addr_res = client.update_address("9876543210", "456 Park Avenue, Mumbai")
    assert addr_res["success"] is True

    # Query DB directly to verify the address update persisted
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT address FROM customers WHERE phone_number = '9876543210'")
    assert cursor.fetchone()["address"] == "456 Park Avenue, Mumbai"
    conn.close()

def test_dialogue_manager_cancel_confirmation_flow():
    dm = DialogueManager()
    
    # Step 1: Request cancel order
    res1 = dm.process_turn("cancel_order", {}, "please cancel order")
    assert dm.sessions["default"]["state"] == DialogueState.AWAITING_ORDER_ID
    assert "What is your 6-digit Order ID" in res1["response"]

    # Step 2: Provide Order ID -> should ask for confirmation
    res2 = dm.process_turn("unknown", {"order_id": "ORD-876543"}, "876543")
    assert dm.sessions["default"]["state"] == DialogueState.AWAITING_CANCEL_CONFIRM
    assert "Are you sure you want to cancel order ORD-876543" in res2["response"]

    # Step 3: Say No (Abort)
    res3 = dm.process_turn("unknown", {}, "no do not cancel")
    assert dm.sessions["default"]["state"] == DialogueState.IDLE
    assert "will not cancel your order" in res3["response"]

    # Order in DB should STILL be Shipped (not Cancelled)
    client = BackendClient()
    assert client.get_order_status("ORD-876543")["status"] == "Shipped"

    # Step 4: Start again, and say Yes (Confirm)
    dm.process_turn("cancel_order", {}, "cancel order")
    dm.process_turn("unknown", {"order_id": "ORD-876543"}, "876543")
    res4 = dm.process_turn("unknown", {}, "yes, confirm cancellation")
    
    assert dm.sessions["default"]["state"] == DialogueState.IDLE
    assert "successfully cancelled your order ORD-876543" in res4["response"]

    # Order in DB should now be Cancelled!
    assert client.get_order_status("ORD-876543")["status"] == "Cancelled"
