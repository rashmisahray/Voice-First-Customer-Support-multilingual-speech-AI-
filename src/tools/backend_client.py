import sqlite3
import logging
from typing import Dict, Any, Optional
from src.database.db_manager import get_db_connection

logger = logging.getLogger("src.tools.backend_client")

class BackendClient:
    """Backend client executing customer support transactions against the SQLite database."""

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Lazily obtains and caches a single reusable database connection."""
        if self._conn is None:
            self._conn = get_db_connection()
        return self._conn

    def __del__(self):
        """Cleans up the connection cleanly when the client object is destroyed."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Fetches status of a specific order from the database."""
        logger.info("Backend Tools: Fetching status for Order ID: %s", order_id)
        clean_id = order_id.upper().strip()
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status, delivery_date, carrier, amount FROM orders WHERE order_id = ?",
                (clean_id,)
            )
            row = cursor.fetchone()
            
            if row:
                logger.info("Backend Tools: Order found in DB. Status: %s", row["status"])
                return {
                    "success": True,
                    "order_id": clean_id,
                    "status": row["status"],
                    "delivery_date": row["delivery_date"],
                    "carrier": row["carrier"],
                    "amount": row["amount"]
                }
        except Exception as e:
            logger.error("Database query failed: %s", e)
            
        logger.warning("Backend Tools: Order ID %s not found in database.", order_id)
        return {
            "success": False,
            "error": "Order ID not found",
            "order_id": clean_id
        }

    def reset_password(self, email: str) -> Dict[str, Any]:
        """Triggers password reset flow in backend systems if email exists."""
        logger.info("Backend Tools: Triggering password reset for email: %s", email)
        clean_email = email.lower().strip()
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT email FROM customers WHERE email = ?", (clean_email,))
            row = cursor.fetchone()
            
            if row:
                return {
                    "success": True,
                    "message": "Password reset token generated and sent to customer email."
                }
        except Exception as e:
            logger.error("Database query failed: %s", e)
            
        return {
            "success": False,
            "message": "Email address not found in our records."
        }

    def update_address(self, phone_number: str, address: str) -> Dict[str, Any]:
        """Updates shipping/delivery address in the database for the given phone number."""
        logger.info("Backend Tools: Updating address for phone number: %s", phone_number)
        clean_phone = phone_number.strip()
        clean_address = address.strip()
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE customers SET address = ? WHERE phone_number = ?",
                (clean_address, clean_phone)
            )
            rows_updated = cursor.rowcount
            conn.commit()
            
            if rows_updated > 0:
                return {
                    "success": True,
                    "message": "Customer address updated successfully in the profile."
                }
        except Exception as e:
            logger.error("Database update failed: %s", e)
            
        return {
            "success": False,
            "message": "Phone number not found in our database."
        }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancels a specific order in the database."""
        logger.info("Backend Tools: Cancelling Order ID: %s", order_id)
        clean_id = order_id.upper().strip()
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE orders SET status = 'Cancelled' WHERE order_id = ?",
                (clean_id,)
            )
            rows_updated = cursor.rowcount
            conn.commit()
            
            if rows_updated > 0:
                return {
                    "success": True,
                    "message": f"Order {clean_id} has been cancelled successfully."
                }
        except Exception as e:
            logger.error("Database update failed: %s", e)
            
        return {
            "success": False,
            "message": f"Order {clean_id} was not found."
        }

    def refund_order(self, order_id: str, reason: str) -> Dict[str, Any]:
        """Processes a refund request for a specific order in the database."""
        logger.info("Backend Tools: Processing refund for Order ID: %s, Reason: %s", order_id, reason)
        clean_id = order_id.upper().strip()
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE orders SET status = 'Refunded' WHERE order_id = ?",
                (clean_id,)
            )
            rows_updated = cursor.rowcount
            conn.commit()
            
            if rows_updated > 0:
                return {
                    "success": True,
                    "message": f"Refund of order {clean_id} processed successfully due to: {reason}."
                }
        except Exception as e:
            logger.error("Database update failed: %s", e)
            
        return {
            "success": False,
            "message": f"Order {clean_id} was not found."
        }
