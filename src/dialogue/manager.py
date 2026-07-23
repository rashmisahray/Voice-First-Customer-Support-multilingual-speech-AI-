import logging
from typing import Dict, Any, Optional
from enum import Enum
from src.core.config import settings
from src.tools.backend_client import BackendClient
from src.nlu.llm_fallback import LLMFallback

logger = logging.getLogger("src.dialogue.manager")

class DialogueState(str, Enum):
    IDLE = "idle"
    AWAITING_ORDER_ID = "awaiting_order_id"
    AWAITING_EMAIL = "awaiting_email"
    AWAITING_PHONE = "awaiting_phone"
    AWAITING_ADDRESS = "awaiting_address"
    AWAITING_REASON = "awaiting_reason"
    AWAITING_CANCEL_CONFIRM = "awaiting_cancel_confirm"
    FALLBACK = "fallback"

class DialogueManager:
    """
    State-machine based Dialogue Manager for Vani.
    Tracks session state, manages slot filling, supports dynamic intent switching, and executes backend tool calls.
    """

    def __init__(self, backend_client: Optional[BackendClient] = None):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.backend_client = backend_client or BackendClient()
        self.llm_fallback = LLMFallback()
        logger.info("Initializing DialogueManager with BackendClient and LLMFallback.")

    def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Retrieves an existing dialogue session or initializes a new state container."""
        if session_id not in self.sessions:
            logger.info("Dialogue Manager: Initializing new dialogue session: %s", session_id)
            self.sessions[session_id] = {
                "state": DialogueState.IDLE,
                "context": {},
                "history": []
            }
        return self.sessions[session_id]

    def reset_session(self, session_id: str):
        """Resets the state and context of a dialogue session."""
        if session_id in self.sessions:
            self.sessions[session_id]["state"] = DialogueState.IDLE
            self.sessions[session_id]["context"].clear()
            self.sessions[session_id]["history"].clear()
            logger.info("Dialogue Manager: Session %s reset to IDLE.", session_id)

    def _reset_workflow_context(self, session: Dict[str, Any]):
        """Resets task-specific variables but keeps persistent user memory (name, email, phone)."""
        context = session["context"]
        for key in ["order_id", "reason", "address", "workflow"]:
            if key in context:
                del context[key]

    def process_turn(
        self, 
        intent: str, 
        entities: Dict[str, Any], 
        text: str, 
        session_id: str = "default"
    ) -> Dict[str, Any]:
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
            if v is not None:
                context[k] = v

        # Determine waiting slot based on current state
        waiting_slot = None
        if current_state == DialogueState.AWAITING_ORDER_ID:
            waiting_slot = "order_id"
        elif current_state == DialogueState.AWAITING_EMAIL:
            waiting_slot = "email"
        elif current_state == DialogueState.AWAITING_PHONE:
            waiting_slot = "phone_number"
        elif current_state == DialogueState.AWAITING_ADDRESS:
            waiting_slot = "address"
        elif current_state == DialogueState.AWAITING_REASON:
            waiting_slot = "reason"
        elif current_state == DialogueState.AWAITING_CANCEL_CONFIRM:
            waiting_slot = "confirmation (yes/no)"

        active_intent = context.get("workflow", "None")

        logger.debug("--- Dialogue Turn Debug ---")
        logger.debug("Incoming transcript: '%s'", text)
        logger.debug("Current session state: %s", current_state)
        logger.debug("Active intent: %s", active_intent)
        logger.debug("Waiting slot: %s", waiting_slot)
        logger.debug("Extracted entities: %s", entities)

        # 2. Global Interruption Check (Greetings or Goodbyes)
        if intent == "greeting":
            session["state"] = DialogueState.IDLE
            self._reset_workflow_context(session)
            response_text = "Hello! Welcome to Vani Customer Support. How can I help you today? You can check order status, reset password, or update address."
            session["history"].append({"speaker": "assistant", "text": response_text})
            return {"response": response_text, "state": session["state"], "tool_executed": None, "tool_result": None}
            
        elif intent == "farewell":
            session["state"] = DialogueState.IDLE
            self._reset_workflow_context(session)
            response_text = "Thank you for choosing Vani Customer Support. Have a wonderful day ahead! Namaste."
            session["history"].append({"speaker": "assistant", "text": response_text})
            return {"response": response_text, "state": session["state"], "tool_executed": None, "tool_result": None}

        # 3. Intent Switching Logic: If user is in an awaiting state but provides a NEW intent or entities matching one
        if current_state != DialogueState.IDLE:
            # Check if entities for a DIFFERENT intent were supplied
            if "email" in entities and current_state != DialogueState.AWAITING_EMAIL:
                intent = "password_reset"
            elif "order_id" in entities and current_state != DialogueState.AWAITING_ORDER_ID:
                expected_workflow = context.get("workflow")
                if expected_workflow not in ["cancel_order", "refund_request", "order_status"]:
                    intent = "order_status"
            elif ("phone_number" in entities or "address" in entities) and current_state not in [DialogueState.AWAITING_PHONE, DialogueState.AWAITING_ADDRESS]:
                intent = "update_address"

            # Determine expected intent/workflow for current state
            expected_intent = None
            if current_state == DialogueState.AWAITING_ORDER_ID:
                expected_intent = context.get("workflow") or "order_status"
            elif current_state == DialogueState.AWAITING_EMAIL:
                expected_intent = "password_reset"
            elif current_state in [DialogueState.AWAITING_PHONE, DialogueState.AWAITING_ADDRESS]:
                expected_intent = "update_address"
            elif current_state == DialogueState.AWAITING_REASON:
                expected_intent = "refund_request"

            # If user explicitly switched intent, reset context for the new workflow
            if intent != "unknown" and expected_intent and intent != expected_intent:
                logger.info("Dialogue Manager: Switching intent from %s to %s", current_state, intent)
                session["state"] = DialogueState.IDLE
                current_state = DialogueState.IDLE
                self._reset_workflow_context(session)
                # Re-apply current entities
                for k, v in entities.items():
                    if v is not None:
                        context[k] = v

        # 4. State Machine Transition Logic
        if current_state == DialogueState.IDLE:
            if intent == "order_status":
                context["workflow"] = "order_status"
                if "order_id" in context:
                    order_id = context["order_id"]
                    tool_executed = "get_order_status"
                    tool_result = self.backend_client.get_order_status(order_id)
                    if tool_result.get("success"):
                        response_text = f"I found your order {order_id}. The status is '{tool_result['status']}', shipped via {tool_result['carrier']}. It is expected on {tool_result['delivery_date']}."
                    else:
                        response_text = f"I'm sorry, I couldn't find an order with ID {order_id} in our records."
                    session["state"] = DialogueState.IDLE
                    self._reset_workflow_context(session)
                else:
                    session["state"] = DialogueState.AWAITING_ORDER_ID
                    response_text = "I'd be happy to check your order status. Could you please state your 6-digit Order ID?"

            elif intent == "cancel_order":
                context["workflow"] = "cancel_order"
                if "order_id" in context:
                    order_id = context["order_id"]
                    session["state"] = DialogueState.AWAITING_CANCEL_CONFIRM
                    response_text = f"Are you sure you want to cancel order {order_id}? Please say yes or confirm."
                else:
                    session["state"] = DialogueState.AWAITING_ORDER_ID
                    response_text = "I can help you cancel your order. What is your 6-digit Order ID?"

            elif intent == "refund_request":
                context["workflow"] = "refund_request"
                if "order_id" not in context:
                    session["state"] = DialogueState.AWAITING_ORDER_ID
                    response_text = "Sure, I can help you with a refund. What is your 6-digit Order ID?"
                elif "reason" not in context:
                    session["state"] = DialogueState.AWAITING_REASON
                    response_text = "Could you please tell me the reason for the refund?"
                else:
                    order_id = context["order_id"]
                    reason = context["reason"]
                    tool_executed = "refund_order"
                    tool_result = self.backend_client.refund_order(order_id, reason)
                    response_text = f"I have successfully processed a refund request for your order {order_id} because the item was {reason}."
                    session["state"] = DialogueState.IDLE
                    self._reset_workflow_context(session)

            elif intent == "password_reset":
                context["workflow"] = "password_reset"
                if "email" in context:
                    email = context["email"]
                    tool_executed = "reset_password"
                    tool_result = self.backend_client.reset_password(email)
                    response_text = f"Done! A password reset token has been generated and sent to {email}. Please check your inbox."
                    session["state"] = DialogueState.IDLE
                    self._reset_workflow_context(session)
                else:
                    session["state"] = DialogueState.AWAITING_EMAIL
                    response_text = "Sure, I can help you reset your password. What is your registered email address?"

            elif intent == "update_address":
                context["workflow"] = "update_address"
                if "phone_number" in context:
                    if "address" in context:
                        phone = context["phone_number"]
                        address = context["address"]
                        tool_executed = "update_address"
                        tool_result = self.backend_client.update_address(phone, address)
                        response_text = f"Thank you. I have successfully updated the delivery address for {phone} to: {address}."
                        session["state"] = DialogueState.IDLE
                        self._reset_workflow_context(session)
                    else:
                        session["state"] = DialogueState.AWAITING_ADDRESS
                        response_text = "I have your phone number. What is the new shipping address you would like to set?"
                else:
                    session["state"] = DialogueState.AWAITING_PHONE
                    response_text = "I can help you update your address. First, could you tell me your 10-digit registered phone number?"

            else:
                # Fallback for unknown intent
                session["state"] = DialogueState.FALLBACK
                if settings.dialogue.enable_llm_fallback:
                    response_text = self.llm_fallback.generate_response(text, session_id)
                else:
                    response_text = "I'm not sure I understand that request. Could you please rephrase, or say 'order status' or 'reset password' if you need help?"
                session["state"] = DialogueState.IDLE

        # State: AWAITING_ORDER_ID
        elif current_state == DialogueState.AWAITING_ORDER_ID:
            if "order_id" in context:
                order_id = context["order_id"]
                workflow = context.get("workflow", "order_status")
                
                if workflow == "cancel_order":
                    session["state"] = DialogueState.AWAITING_CANCEL_CONFIRM
                    response_text = f"Are you sure you want to cancel order {order_id}? Please say yes or confirm."
                elif workflow == "refund_request":
                    if "reason" in context:
                        reason = context["reason"]
                        tool_executed = "refund_order"
                        tool_result = self.backend_client.refund_order(order_id, reason)
                        response_text = f"I have successfully processed a refund request for your order {order_id} because the item was {reason}."
                        session["state"] = DialogueState.IDLE
                        self._reset_workflow_context(session)
                    else:
                        session["state"] = DialogueState.AWAITING_REASON
                        response_text = "Could you please tell me the reason for the refund?"
                else:
                    tool_executed = "get_order_status"
                    tool_result = self.backend_client.get_order_status(order_id)
                    if tool_result.get("success"):
                        response_text = f"Got it. Order {order_id} is '{tool_result['status']}', handled by {tool_result['carrier']}. Delivery is scheduled for {tool_result['delivery_date']}."
                    else:
                        response_text = f"I'm sorry, Order ID {order_id} was not found in our records."
                    session["state"] = DialogueState.IDLE
                    self._reset_workflow_context(session)
            else:
                response_text = "I didn't hear a valid 6-digit Order ID. Could you please state your Order ID?"

        # State: AWAITING_REASON
        elif current_state == DialogueState.AWAITING_REASON:
            reason = context.get("reason", text).strip()
            if reason:
                context["reason"] = reason
                order_id = context.get("order_id")
                
                if order_id:
                    tool_executed = "refund_order"
                    tool_result = self.backend_client.refund_order(order_id, reason)
                    response_text = f"I have successfully processed a refund request for your order {order_id} because the item was {reason}."
                    session["state"] = DialogueState.IDLE
                    self._reset_workflow_context(session)
                else:
                    session["state"] = DialogueState.AWAITING_ORDER_ID
                    response_text = "I have the reason. Now, what is the 6-digit Order ID?"
            else:
                response_text = "Could you please tell me the reason for the refund?"

        # State: AWAITING_CANCEL_CONFIRM
        elif current_state == DialogueState.AWAITING_CANCEL_CONFIRM:
            clean_text = text.lower().strip()
            order_id = context.get("order_id")
            
            # Simple keyword check for yes/confirm
            positive_words = ["yes", "confirm", "haan", "ha", "sure", "okay", "ok", "हाँ", "हा"]
            negative_words = ["no", "abort", "cancel", "nahi", "na", "ना", "नही"]
            
            is_positive = any(w in clean_text for w in positive_words)
            is_negative = any(w in clean_text for w in negative_words)
            
            if is_positive:
                if order_id:
                    tool_executed = "cancel_order"
                    tool_result = self.backend_client.cancel_order(order_id)
                    if tool_result.get("success"):
                        response_text = f"I have successfully cancelled your order {order_id}."
                    else:
                        response_text = f"I'm sorry, I failed to cancel order {order_id} because it was not found."
                else:
                    response_text = "I don't have a valid Order ID to cancel."
                session["state"] = DialogueState.IDLE
                self._reset_workflow_context(session)
            elif is_negative:
                response_text = "Okay, I will not cancel your order. How else can I help you today?"
                session["state"] = DialogueState.IDLE
                self._reset_workflow_context(session)
            else:
                response_text = f"Please say yes to confirm cancellation of order {order_id}, or say no to abort."

        # State: AWAITING_EMAIL
        elif current_state == DialogueState.AWAITING_EMAIL:
            if "email" in context:
                email = context["email"]
                tool_executed = "reset_password"
                tool_result = self.backend_client.reset_password(email)
                response_text = f"A password reset link has been dispatched to {email}. Please verify your inbox."
                session["state"] = DialogueState.IDLE
                self._reset_workflow_context(session)
            else:
                response_text = "I couldn't catch a valid email address. What is your registered email?"

        # State: AWAITING_PHONE
        elif current_state == DialogueState.AWAITING_PHONE:
            if "phone_number" in context:
                if "address" in context:
                    phone = context["phone_number"]
                    address = context["address"]
                    tool_executed = "update_address"
                    tool_result = self.backend_client.update_address(phone, address)
                    response_text = f"Successfully updated your delivery address to: {address}."
                    session["state"] = DialogueState.IDLE
                    self._reset_workflow_context(session)
                else:
                    session["state"] = DialogueState.AWAITING_ADDRESS
                    response_text = "Thanks. Now, please tell me the new delivery address."
            else:
                response_text = "I need your 10-digit registered phone number to proceed. Could you tell me your phone number?"

        # State: AWAITING_ADDRESS
        elif current_state == DialogueState.AWAITING_ADDRESS:
            address = context.get("address", text).strip()
            if address:
                context["address"] = address
                phone = context.get("phone_number", "9876543210")
                tool_executed = "update_address"
                tool_result = self.backend_client.update_address(phone, address)
                response_text = f"Successfully updated your address to: '{address}'."
                session["state"] = DialogueState.IDLE
                self._reset_workflow_context(session)
            else:
                response_text = "Please tell me the new delivery address."

        else:
            response_text = "How else can I help you today?"
            session["state"] = DialogueState.IDLE
            self._reset_workflow_context(session)

        session["history"].append({"speaker": "assistant", "text": response_text})
        
        next_state = session["state"]
        logger.debug("State transition: %s -> %s", current_state, next_state)
        logger.debug("---------------------------")
        
        return {
            "response": response_text, 
            "state": session["state"],
            "tool_executed": tool_executed,
            "tool_result": tool_result
        }
