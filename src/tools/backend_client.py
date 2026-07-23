import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("src.tools.backend_client")

# Mock database representing backend records
MOCK_ORDERS = {
    "ORD-876543": {
        "status": "Shipped",
        "delivery_date": "2026-07-06",
        "carrier": "Blue Dart",
        "amount": 2499.00
    },
    "ORD-123456": {
        "status": "Delivered",
        "delivery_date": "2026-07-01",
        "carrier": "Delhivery",
        "amount": 899.00
    }
}

class BackendClient:
    """Stubbed backend client for handling external customer support operations."""

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Fetches status of a specific order. Never hallucinates responses.
        
        Args:
            order_id: Order identification string.
            
        Returns:
            Dict containing order details or error code.
        """
        logger.info("Backend Tools: Fetching status for Order ID: %s", order_id)
        
        # Clean the input format to match database keys
        clean_id = order_id.upper().strip()
        
        if clean_id in MOCK_ORDERS:
            order_info = MOCK_ORDERS[clean_id]
            logger.info("Backend Tools: Order found. Status: %s", order_info["status"])
            return {
                "success": True,
                "order_id": clean_id,
                **order_info
            }
        
        logger.warning("Backend Tools: Order ID %s not found in records.", order_id)
        return {
            "success": False,
            "error": "Order ID not found",
            "order_id": clean_id
        }

    def reset_password(self, email: str) -> Dict[str, Any]:
        """Triggers password reset flow in backend systems.
        
        Args:
            email: User's registered email address.
            
        Returns:
            Dict detailing reset status.
        """
        logger.info("Backend Tools: Triggering password reset for email: %s", email)
        return {
            "success": True,
            "message": "Password reset token generated and sent to customer email."
        }

    def update_address(self, phone_number: str, address: str) -> Dict[str, Any]:
        """Updates shipping/delivery address in backend systems.
        
        Args:
            phone_number: Customer's phone number.
            address: New text address details.
            
        Returns:
            Dict detailing address update status.
        """
        logger.info("Backend Tools: Updating address for phone number: %s", phone_number)
        return {
            "success": True,
            "message": "Customer address updated successfully in the profile."
        }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancels a specific order in backend systems.
        
        Args:
            order_id: Order identification string.
            
        Returns:
            Dict detailing cancellation status.
        """
        logger.info("Backend Tools: Cancelling Order ID: %s", order_id)
        return {
            "success": True,
            "message": f"Order {order_id} has been cancelled successfully."
        }

    def refund_order(self, order_id: str, reason: str) -> Dict[str, Any]:
        """Processes a refund request for a specific order in backend systems.
        
        Args:
            order_id: Order identification string.
            reason: Reason for refund.
            
        Returns:
            Dict detailing refund status.
        """
        logger.info("Backend Tools: Processing refund for Order ID: %s, Reason: %s", order_id, reason)
        return {
            "success": True,
            "message": f"Refund of order {order_id} processed successfully due to: {reason}."
        }
