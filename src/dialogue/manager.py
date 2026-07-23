import os
import json
import logging
import httpx
from typing import Dict, Any, List, Optional
from enum import Enum
from src.core.config import settings
from src.tools.backend_client import BackendClient

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

SYSTEM_PROMPT = """You are Vani, a helpful multilingual customer support voice assistant for an e-commerce platform.
Your job is to orchestrate the conversation, determine user intent, collect missing parameters (slots) for tools, and decide when to call backend tools.

Available tools:
1. get_order_status(order_id: string): Checks status of an order. order_id must be exactly in format "ORD-XXXXXX" (6 digits).
2. reset_password(email: string): Sends a password reset link to the given email address.
3. update_address(phone_number: string, address: string): Updates delivery address. phone_number must be exactly 10 digits.
4. cancel_order(order_id: string): Cancels an active order. order_id format "ORD-XXXXXX".
5. refund_order(order_id: string, reason: string): Processes refund. order_id format "ORD-XXXXXX".

Guidelines:
- If required slots are missing for the intent, identify them in "missing_slots" and ask the user for ONLY the first missing slot in "assistant_reply".
- If all slots are present, set "tool" to the tool name and fill "entities". Leave "missing_slots" empty.
- Keep "assistant_reply" concise (1-2 sentences), conversational, and in the language the user is speaking (English, Hindi, or Hinglish).
- Always return ONLY a valid JSON object matching this schema:
{
  "intent": "greeting" | "farewell" | "order_status" | "password_reset" | "update_address" | "cancel_order" | "refund_request" | "unknown",
  "tool": "get_order_status" | "reset_password" | "update_address" | "cancel_order" | "refund_order" | null,
  "entities": {
    "order_id": string | null,
    "email": string | null,
    "phone_number": string | null,
    "address": string | null,
    "reason": string | null
  },
  "missing_slots": string[],
  "assistant_reply": string
}
"""

class DialogueManager:
    """
    Gemini-powered Dialogue Manager for Vani.
    Uses the Gemini API to orchestrate customer support turns, execute tools, 
    and synthesize voice-friendly natural language responses.
    """

    def __init__(self, backend_client: Optional[BackendClient] = None):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.backend_client = backend_client or BackendClient()
        logger.info("Initializing Gemini DialogueManager.")

    def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Retrieves or initializes session storage."""
        if session_id not in self.sessions:
            logger.info("Dialogue Manager: Creating session %s", session_id)
            self.sessions[session_id] = {
                "state": DialogueState.IDLE,
                "context": {},
                "history": []  # List of dicts: {"role": "user"|"model", "text": str}
            }
        return self.sessions[session_id]

    def reset_session(self, session_id: str):
        """Resets dialogue session context and history."""
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
        
        logger.info("Dialogue Manager (%s): Processing turn. State: %s -> User Input: '%s'", 
                    session_id, current_state, text)
        
        # Append user turn to history
        session["history"].append({"role": "user", "text": text})
        
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if api_key:
            return self._process_gemini_turn(text, session, api_key)
        else:
            return self._process_fallback_turn(intent, entities, text, session)

    def _process_gemini_turn(self, text: str, session: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        """Orchestrates dialogue using the live Gemini API."""
        # Prune conversation history to the last 10 turns to minimize payload token size and latency
        if len(session["history"]) > 10:
            session["history"] = session["history"][-10:]

        # 1. Build conversational history payload
        contents = []
        for turn in session["history"][:-1]:  # Exclude current user message to avoid duplicate
            contents.append({
                "role": "user" if turn["role"] == "user" else "model",
                "parts": [{"text": turn["text"]}]
            })
        # Add current user turn
        contents.append({
            "role": "user",
            "parts": [{"text": text}]
        })

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
        payload = {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2
            }
        }
        
        tool_executed = None
        tool_result = None
        response_text = ""
        
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    result_data = resp.json()
                    content = result_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    gemini_res = json.loads(content)
                    
                    logger.debug("Gemini Orchestration JSON output: %s", gemini_res)
                    
                    tool_name = gemini_res.get("tool")
                    extracted_entities = gemini_res.get("entities", {})
                    missing_slots = gemini_res.get("missing_slots", [])
                    response_text = gemini_res.get("assistant_reply", "")
                    
                    # 2. Check if we should execute a tool call
                    if tool_name and not missing_slots:
                        tool_executed = tool_name
                        tool_result = self._dispatch_tool(tool_name, extracted_entities)
                        
                        # 3. Second call to Gemini: Synthesize final response based on tool results
                        response_text = self._synthesize_final_reply(text, tool_name, tool_result, api_key)
                    
                    # Map state dynamically based on missing slots
                    next_state = self._map_state_from_slots(missing_slots, gemini_res.get("intent"))
                    session["state"] = next_state
                    
                else:
                    logger.error("Gemini API returned error code %d: %s", resp.status_code, resp.text)
                    response_text = "I'm sorry, I encountered a temporary connection issue. How can I help you?"
                    session["state"] = DialogueState.IDLE
        except Exception as e:
            logger.error("Gemini turn processing failed: %s", e)
            response_text = "I'm sorry, I had trouble processing that request. Could you say it again?"
            session["state"] = DialogueState.IDLE

        # Save assistant turn to history
        session["history"].append({"role": "model", "text": response_text})
        
        return {
            "response": response_text,
            "state": session["state"],
            "tool_executed": tool_executed,
            "tool_result": tool_result
        }

    def _synthesize_final_reply(self, user_query: str, tool_name: str, tool_result: Dict[str, Any], api_key: str) -> str:
        """Invokes Gemini a second time to translate tool outcome into a natural response."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
        prompt = (
            f"The user query was: '{user_query}'\n"
            f"The backend tool '{tool_name}' was executed successfully.\n"
            f"Tool Execution Result: {json.dumps(tool_result)}\n\n"
            "Based on this result, generate a concise, friendly, and natural voice response (1-2 sentences) "
            "for the customer. Keep the same language (English, Hindi, or Hinglish) they used. Do not include JSON formatting."
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3}
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            logger.error("Second-pass reply synthesis failed: %s", e)
        
        # Safe string fallback if Gemini synthesis fails
        return tool_result.get("message", "Request processed successfully.")

    def _dispatch_tool(self, tool_name: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Invokes the actual backend client method for the matched tool."""
        logger.info("Executing backend tool: %s with entities: %s", tool_name, entities)
        if tool_name == "get_order_status":
            return self.backend_client.get_order_status(entities.get("order_id"))
        elif tool_name == "reset_password":
            return self.backend_client.reset_password(entities.get("email"))
        elif tool_name == "update_address":
            return self.backend_client.update_address(entities.get("phone_number"), entities.get("address"))
        elif tool_name == "cancel_order":
            return self.backend_client.cancel_order(entities.get("order_id"))
        elif tool_name == "refund_order":
            return self.backend_client.refund_order(entities.get("order_id"), entities.get("reason"))
        return {"success": False, "error": "Unknown tool"}

    def _map_state_from_slots(self, missing_slots: List[str], intent: str) -> DialogueState:
        """Maps missing slot lists to visualizer-compatible DialogueStates."""
        if not missing_slots:
            if intent == "cancel_order":
                # For cancel_order, we might transition to confirmation
                return DialogueState.IDLE
            return DialogueState.IDLE
            
        slot = missing_slots[0]
        if slot == "order_id":
            return DialogueState.AWAITING_ORDER_ID
        elif slot == "email":
            return DialogueState.AWAITING_EMAIL
        elif slot == "phone_number":
            return DialogueState.AWAITING_PHONE
        elif slot == "address":
            return DialogueState.AWAITING_ADDRESS
        elif slot == "reason":
            return DialogueState.AWAITING_REASON
        return DialogueState.IDLE

    def _process_fallback_turn(self, intent: str, entities: Dict[str, Any], text: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Zero-key local state machine fallback preserving full test compatibility."""
        context = session["context"]
        current_state = session["state"]
        
        # Merge new entities into context
        for k, v in entities.items():
            if v is not None:
                context[k] = v

        # Interruption/intent switching check
        if current_state != DialogueState.IDLE and intent not in ["unknown", context.get("workflow")]:
            self._reset_workflow_context(session)
            session["state"] = DialogueState.IDLE
            current_state = DialogueState.IDLE

        response_text = ""
        tool_executed = None
        tool_result = None

        # Standard greetings / goodbye overrides
        if intent == "greeting":
            session["state"] = DialogueState.IDLE
            self._reset_workflow_context(session)
            response_text = "Hello! Welcome to Vani Customer Support. How can I help you today? You can check order status, reset password, or update address."
            session["history"].append({"role": "model", "text": response_text})
            return {"response": response_text, "state": session["state"], "tool_executed": None, "tool_result": None}
            
        elif intent == "farewell":
            session["state"] = DialogueState.IDLE
            self._reset_workflow_context(session)
            response_text = "Thank you for choosing Vani Customer Support. Have a wonderful day ahead! Namaste."
            session["history"].append({"role": "model", "text": response_text})
            return {"response": response_text, "state": session["state"], "tool_executed": None, "tool_result": None}

        # Multi-turn slot logic simulator
        if current_state == DialogueState.IDLE:
            if intent == "order_status":
                context["workflow"] = "order_status"
                if "order_id" in context:
                    order_id = context["order_id"]
                    tool_executed = "get_order_status"
                    tool_result = self.backend_client.get_order_status(order_id)
                    response_text = f"I found your order {order_id}. The status is '{tool_result.get('status')}', shipped via {tool_result.get('carrier')}."
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
                    response_text = f"Done! A password reset token has been generated and sent to {email}."
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
                response_text = "How else can I help you today?"
                session["state"] = DialogueState.IDLE

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
                        response_text = f"I have processed a refund request for order {order_id}."
                        session["state"] = DialogueState.IDLE
                        self._reset_workflow_context(session)
                    else:
                        session["state"] = DialogueState.AWAITING_REASON
                        response_text = "Could you please tell me the reason for the refund?"
                else:
                    tool_executed = "get_order_status"
                    tool_result = self.backend_client.get_order_status(order_id)
                    response_text = f"Got it. Order {order_id} status is {tool_result.get('status')}."
                    session["state"] = DialogueState.IDLE
                    self._reset_workflow_context(session)
            else:
                response_text = "I didn't hear a valid 6-digit Order ID. Could you please state your Order ID?"

        elif current_state == DialogueState.AWAITING_CANCEL_CONFIRM:
            clean = text.lower().strip()
            order_id = context.get("order_id")
            if any(w in clean for w in ["yes", "confirm", "haan", "ha"]):
                tool_executed = "cancel_order"
                tool_result = self.backend_client.cancel_order(order_id)
                response_text = f"I have successfully cancelled your order {order_id}."
                session["state"] = DialogueState.IDLE
                self._reset_workflow_context(session)
            else:
                response_text = "Okay, I will not cancel your order."
                session["state"] = DialogueState.IDLE
                self._reset_workflow_context(session)

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

        elif current_state == DialogueState.AWAITING_EMAIL:
            if "email" in context:
                email = context["email"]
                tool_executed = "reset_password"
                tool_result = self.backend_client.reset_password(email)
                response_text = f"A password reset link has been dispatched to {email}."
                session["state"] = DialogueState.IDLE
                self._reset_workflow_context(session)
            else:
                response_text = "I couldn't catch a valid email address. What is your registered email?"

        elif current_state == DialogueState.AWAITING_PHONE:
            if "phone_number" in context:
                session["state"] = DialogueState.AWAITING_ADDRESS
                response_text = "Thanks. Now, please tell me the new delivery address."
            else:
                response_text = "I need your 10-digit registered phone number to proceed."

        elif current_state == DialogueState.AWAITING_ADDRESS:
            if "address" in context:
                phone = context.get("phone_number", "9876543210")
                address = context["address"]
                tool_executed = "update_address"
                tool_result = self.backend_client.update_address(phone, address)
                response_text = f"Successfully updated your address to: '{address}'."
                session["state"] = DialogueState.IDLE
                self._reset_workflow_context(session)
            else:
                response_text = "Please tell me the new delivery address."

        else:
            response_text = "I'm here to help."
            session["state"] = DialogueState.IDLE

        session["history"].append({"role": "model", "text": response_text})
        return {
            "response": response_text,
            "state": session["state"],
            "tool_executed": tool_executed,
            "tool_result": tool_result
        }
