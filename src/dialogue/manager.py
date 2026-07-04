import logging
from enum import Enum
from typing import Dict, Any, List
from src.core.config import settings

logger = logging.getLogger("src.dialogue.manager")

class DialogueState(str, Enum):
    IDLE = "idle"
    AWAITING_ORDER_ID = "awaiting_order_id"
    PROCESSING_ORDER = "processing_order"
    COMPLETED = "completed"
    FALLBACK = "fallback"

class DialogueManager:
    """Tracks session conversation states and handles state transitions."""

    def __init__(self):
        self.state = DialogueState.IDLE
        self.history: List[Dict[str, str]] = []
        self.context: Dict[str, Any] = {}

    def process_turn(self, intent: str, entities: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Processes a single conversational turn.
        
        Args:
            intent: Identified intent.
            entities: Extracted entities.
            text: Raw transcript.
            
        Returns:
            Dict containing the dialogue response and next state.
        """
        logger.info("Dialogue Manager: Processing turn. Current State: %s, Intent: %s", self.state, intent)
        
        # Save transcript to history
        self.history.append({"speaker": "user", "text": text})

        response_text = ""
        
        # Handle fallback for unknown intents
        if intent == "unknown":
            self.state = DialogueState.FALLBACK
            if settings.dialogue.enable_llm_fallback:
                logger.info("Dialogue Manager: Triggering LLM Fallback for unknown intent.")
                response_text = "I'm sorry, I couldn't fully understand that. Let me look up options or connect you to an agent."
            else:
                response_text = "Sorry, I am unable to help with that request right now."
            
            self.history.append({"speaker": "assistant", "text": response_text})
            return {"response": response_text, "state": self.state}

        # Safe State Transition Logic
        if self.state == DialogueState.IDLE:
            if intent == "order_status":
                if "order_id" in entities:
                    self.context["order_id"] = entities["order_id"]
                    self.state = DialogueState.PROCESSING_ORDER
                    response_text = f"Got it! Let me check the status for Order ID {entities['order_id']}."
                else:
                    self.state = DialogueState.AWAITING_ORDER_ID
                    response_text = "Sure, I can help you with your order status. Please tell me your 6-digit Order ID."
            else:
                response_text = "Namaste! Welcome to customer support. How can I help you today?"
                
        elif self.state == DialogueState.AWAITING_ORDER_ID:
            if "order_id" in entities:
                self.context["order_id"] = entities["order_id"]
                self.state = DialogueState.PROCESSING_ORDER
                response_text = f"Thanks for sharing the Order ID. I am looking up the status of {entities['order_id']}."
            else:
                response_text = "I didn't catch a valid Order ID. Please provide your Order ID (for example, ORD-876543)."

        elif self.state == DialogueState.PROCESSING_ORDER:
            # Stays in processing or transitions to completed
            self.state = DialogueState.COMPLETED
            response_text = "Your order is currently out for delivery and will arrive shortly."

        else:
            response_text = "Is there anything else I can assist you with?"
            self.state = DialogueState.IDLE

        self.history.append({"speaker": "assistant", "text": response_text})
        logger.info("Dialogue Manager: Transitioned to State: %s, Response: %s", self.state, response_text)
        return {"response": response_text, "state": self.state}
