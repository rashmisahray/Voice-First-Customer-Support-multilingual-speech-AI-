import os
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger("src.database.db_manager")

# Path to the SQLite database file
DB_DIR = Path("data")
DB_PATH = DB_DIR / "vani.db"

def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database, automatically initializing if needed."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    db_existed = DB_PATH.exists()
    
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    
    # Double check if tables actually exist, just in case of empty files
    if not db_existed:
        _init_db_with_conn(conn)
    else:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='customers'")
            if not cursor.fetchone():
                _init_db_with_conn(conn)
        except Exception:
            _init_db_with_conn(conn)
            
    return conn

def _init_db_with_conn(conn: sqlite3.Connection):
    """Initializes tables and seeds data using an active connection."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            phone_number TEXT PRIMARY KEY,
            customer_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            address TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            delivery_date TEXT NOT NULL,
            carrier TEXT NOT NULL,
            amount REAL NOT NULL,
            customer_phone TEXT NOT NULL,
            FOREIGN KEY (customer_phone) REFERENCES customers (phone_number)
        )
    """)
    conn.commit()

    # Seed data
    cursor.execute("SELECT COUNT(*) FROM customers")
    if cursor.fetchone()[0] == 0:
        customers_data = [
            ("9876543210", "John Doe", "john@example.com", "123 Main St, Mumbai"),
            ("9988776655", "Amit Kumar", "amit@gmail.com", "456 Ring Rd, Delhi")
        ]
        cursor.executemany(
            "INSERT INTO customers (phone_number, customer_name, email, address) VALUES (?, ?, ?, ?)",
            customers_data
        )

        orders_data = [
            ("ORD-876543", "Shipped", "2026-07-06", "Blue Dart", 2499.00, "9876543210"),
            ("ORD-123456", "Delivered", "2026-07-01", "Delhivery", 899.00, "9988776655")
        ]
        cursor.executemany(
            "INSERT INTO orders (order_id, status, delivery_date, carrier, amount, customer_phone) VALUES (?, ?, ?, ?, ?, ?)",
            orders_data
        )
        conn.commit()

def init_db():
    """Wrapper to initialize and seed database (compatible with startup hook)."""
    conn = get_db_connection()
    conn.close()

def reset_db():
    """Drops and re-creates database tables with fresh seed data (useful for test isolation)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS orders")
    cursor.execute("DROP TABLE IF EXISTS customers")
    conn.commit()
    _init_db_with_conn(conn)
    conn.close()
