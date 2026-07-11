import logging
from enum import Enum
from typing import Dict, Any, List, Tuple
from src.core.config import settings
from src.tools.backend_client import BackendClient

logger = logging.getLogger("src.dialogue.manager")

class DialogueState(str, Enum):
    IDLE = "idle"
    AWAITING_ORDER_ID = "awaiting_order_id"
    AWAITING_EMAIL = "awaiting_email"
    AWAITING_PHONE = "awaiting_phone"
    AWAITING_ADDRESS = "awaiting_address"
    COMPLETED = "completed"
    FALLBACK = "fallback"

class DialogueManager:
    """
    Dialogue Manager state machine.
    Manages session states, prompts for missing slot values, and triggers backend tools.
    """

    def __init__(self):
        # Dictionary mapping session_id -> session_data
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.backend_client = BackendClient()

    def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            logger.info("Dialogue Manager: Initializing new dialogue session: %s", session_id)
            self.sessions[session_id] = {
                "state": DialogueState.IDLE,
                "history": [],
                "context": {}
            }
        return self.sessions[session_id]

    def reset_session(self, session_id: str):
        """Resets the state and context of a specific session."""
        if session_id in self.sessions:
            self.sessions[session_id] = {
                "state": DialogueState.IDLE,
                "history": [],
                "context": {}
            }
            logger.info("Dialogue Manager: Session %s has been reset.", session_id)

    def process_turn(
        self, 
        intent: str, 
        entities: Dict[str, Any], 
        text: str, 
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Processes a single conversation turn for a session.
        
        Args:
            intent: Classified intent name.
            entities: Dictionary of extracted entities.
            text: Raw transcript text.
            session_id: Unique session identifier.
            
        Returns:
            Dict containing the dialogue response text, next state, and details of any executed tools.
        """
        session = self._get_or_create_session(session_id)
        current_state = session["state"]
        context = session["context"]
        
        logger.info(
            "Dialogue Manager (%s): Processing turn. State: %s -> Intent: %s, Entities: %s", 
            session_id, current_state, intent, entities
        )
        
        # Log user text in session history
        session["history"].append({"speaker": "user", "text": text})

        response_text = ""
        tool_executed = None
        tool_result = None

        # 1. Update session context with newly extracted entities
        for k, v in entities.items():
            context[k] = v

        # 2. Check for Global Intent Interruptions (e.g. Greeting or Goodbye)
        if intent == "greeting":
            session["state"] = DialogueState.IDLE
            context.clear()
            response_text = "Hello! Welcome to Vani Customer Support. How can I help you today? You can check order status, reset password, or update address."
            session["history"].append({"speaker": "assistant", "text": response_text})
            return {"response": response_text, "state": session["state"]}
            
        elif intent == "farewell":
            session["state"] = DialogueState.IDLE
            context.clear()
            response_text = "Thank you for choosing Vani Customer Support. Have a wonderful day ahead! Namaste."
            session["history"].append({"speaker": "assistant", "text": response_text})
            return {"response": response_text, "state": session["state"]}

        # 3. State Machine Transition Logic
        if current_state == DialogueState.IDLE:
            if intent == "order_status":
                if "order_id" in context:
                    # Slot filled. Call backend tool.
                    order_id = context["order_id"]
                    tool_executed = "get_order_status"
                    tool_result = self.backend_client.get_order_status(order_id)
                    
                    if tool_result.get("success"):
                        response_text = f"I found your order {order_id}. The status is '{tool_result['status']}', shipped via {tool_result['carrier']}. It is expected on {tool_result['delivery_date']}."
                    else:
                        response_text = f"I'm sorry, I couldn't find an order with ID {order_id} in our records."
                    session["state"] = DialogueState.IDLE
                    context.clear()
                else:
                    # Missing Order ID slot
                    session["state"] = DialogueState.AWAITING_ORDER_ID
                    response_text = "I'd be happy to check your order status. Could you please provide your 6-digit Order ID?"

            elif intent == "password_reset":
                if "email" in context:
                    # Slot filled. Call backend tool.
                    email = context["email"]
                    tool_executed = "reset_password"
                    tool_result = self.backend_client.reset_password(email)
                    response_text = f"Done! A password reset token has been generated and sent to {email}. Please check your inbox."
                    session["state"] = DialogueState.IDLE
                    context.clear()
                else:
                    # Missing Email slot
                    session["state"] = DialogueState.AWAITING_EMAIL
                    response_text = "Sure, I can help you reset your password. What is your registered email address?"

            elif intent == "update_address":
                if "phone_number" in context:
                    if "address" in context:
                        # Both slots filled. Call backend tool.
                        phone = context["phone_number"]
                        address = context["address"]
                        tool_executed = "update_address"
                        tool_result = self.backend_client.update_address(phone, address)
                        response_text = f"Thank you. I have successfully updated the delivery address for {phone} to: {address}."
                        session["state"] = DialogueState.IDLE
                        context.clear()
                    else:
                        # Missing Address slot
                        session["state"] = DialogueState.AWAITING_ADDRESS
                        response_text = "I have your phone number. What is the new shipping address you would like to set?"
                else:
                    # Missing Phone slot
                    session["state"] = DialogueState.AWAITING_PHONE
                    response_text = "I can help you update your address. First, could you tell me your 10-digit registered phone number?"

            else:
                # Handle unknown intents or LLM fallback
                session["state"] = DialogueState.FALLBACK
                if settings.dialogue.enable_llm_fallback:
                    response_text = "I'm not sure I understand that request. Could you please rephrase, or say 'order status' if you need help with your order?"
                else:
                    response_text = "I am sorry, I can only assist with order status, address updates, and password resets at the moment."
                session["state"] = DialogueState.IDLE

        # State: AWAITING_ORDER_ID
        elif current_state == DialogueState.AWAITING_ORDER_ID:
            if "order_id" in context:
                order_id = context["order_id"]
                tool_executed = "get_order_status"
                tool_result = self.backend_client.get_order_status(order_id)
                if tool_result.get("success"):
                    response_text = f"Got it. Order {order_id} is '{tool_result['status']}', handled by {tool_result['carrier']}. Delivery is scheduled for {tool_result['delivery_date']}."
                else:
                    response_text = f"I'm sorry, Order ID {order_id} was not found in our records."
                session["state"] = DialogueState.IDLE
                context.clear()
            else:
                response_text = "I didn't hear a valid 6-digit Order ID. Could you please state your Order ID?"

        # State: AWAITING_EMAIL
        elif current_state == DialogueState.AWAITING_EMAIL:
            if "email" in context:
                email = context["email"]
                tool_executed = "reset_password"
                tool_result = self.backend_client.reset_password(email)
                response_text = f"A password reset link has been dispatched to {email}. Please verify your inbox."
                session["state"] = DialogueState.IDLE
                context.clear()
            else:
                response_text = "I couldn't catch a valid email address. What is your registered email?"

        # State: AWAITING_PHONE
        elif current_state == DialogueState.AWAITING_PHONE:
            if "phone_number" in context:
                session["state"] = DialogueState.AWAITING_ADDRESS
                response_text = "Thanks. Now, please tell me the new delivery address."
            else:
                response_text = "I need your 10-digit registered phone number to proceed. Could you tell me your phone number?"

        # State: AWAITING_ADDRESS
        elif current_state == DialogueState.AWAITING_ADDRESS:
            # Address can be freeform, if not explicitly parsed by regex, we take the whole user input
            address = context.get("address", text).strip()
            if address:
                context["address"] = address
                phone = context.get("phone_number", "Unknown")
                tool_executed = "update_address"
                tool_result = self.backend_client.update_address(phone, address)
                response_text = f"Successfully updated your address to: '{address}'."
                session["state"] = DialogueState.IDLE
                context.clear()
            else:
                response_text = "Please tell me the new delivery address."

        else:
            response_text = "How else can I help you today?"
            session["state"] = DialogueState.IDLE
            context.clear()

        # Log assistant response in session history
        session["history"].append({"speaker": "assistant", "text": response_text})
        
        logger.info(
            "Dialogue Manager (%s): Turn completed. Next State: %s. Response: '%s'",
            session_id, session["state"], response_text
        )
        
        return {
            "response": response_text,
            "state": session["state"],
            "tool_executed": tool_executed,
            "tool_result": tool_result
        }
