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
- Keep "assistant_reply" concise (1-2 sentences), conversational, and strictly in the customer's language (English, Hindi, or Hinglish).
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
    and synthesize voice-friendly natural language responses in Hindi and English.
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
                "history": [],
                "language": "en"
            }
        return self.sessions[session_id]

    def reset_session(self, session_id: str):
        """Resets dialogue session context and history."""
        if session_id in self.sessions:
            self.sessions[session_id]["state"] = DialogueState.IDLE
            self.sessions[session_id]["context"].clear()
            self.sessions[session_id]["history"].clear()
            self.sessions[session_id]["language"] = "en"
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
        session_id: str = "default",
        language: str = "en"
    ) -> Dict[str, Any]:
        session = self._get_or_create_session(session_id)
        if language and language != "unknown":
            session["language"] = language
            
        current_state = session["state"]
        logger.info("Dialogue Manager (%s): Processing turn. State: %s -> User Input: '%s' (Lang: %s)", 
                    session_id, current_state, text, session.get("language", "en"))
        
        # Append user turn to history
        session["history"].append({"role": "user", "text": text})
        
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if api_key:
            return self._process_gemini_turn(intent, entities, text, session, api_key)
        else:
            return self._process_fallback_turn(intent, entities, text, session)

    def _process_gemini_turn(self, intent: str, entities: Dict[str, Any], text: str, session: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        """Orchestrates dialogue using the live Gemini API with self-healing fallback."""
        if len(session["history"]) > 10:
            session["history"] = session["history"][-10:]

        current_lang = session.get("language", "en")
        
        contents = []
        for turn in session["history"][:-1]:
            contents.append({
                "role": "user" if turn["role"] == "user" else "model",
                "parts": [{"text": turn["text"]}]
            })
        contents.append({
            "role": "user",
            "parts": [{"text": text}]
        })

        system_instruction_text = SYSTEM_PROMPT
        if current_lang in ["hi", "hindi"]:
            system_instruction_text += "\nIMPORTANT: The customer's preferred conversation language is HINDI. Ensure ALL assistant_reply strings are written natively in HINDI (हिंदी script or Hinglish)."
        elif current_lang in ["en", "english"]:
            system_instruction_text += "\nIMPORTANT: The customer's preferred conversation language is ENGLISH. Ensure ALL assistant_reply strings are written natively in ENGLISH."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
        payload = {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": system_instruction_text}]
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
            with httpx.Client(timeout=30.0) as client:
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
                    
                    # Execute tool if ready
                    if tool_name and not missing_slots:
                        tool_executed = tool_name
                        tool_result = self._dispatch_tool(tool_name, extracted_entities, session.get("context"))
                        
                        try:
                            response_text = self._synthesize_final_reply(text, tool_name, tool_result, api_key, language=current_lang)
                        except Exception as synth_err:
                            logger.warning("Gemini second pass failed: %s. Using tool formatter.", synth_err)
                            response_text = self._format_tool_response(tool_name, tool_result, language=current_lang)
                    
                    next_state = self._map_state_from_slots(missing_slots, gemini_res.get("intent"))
                    session["state"] = next_state
                    
                else:
                    logger.error("Gemini API returned error code %d: %s. Falling back to local state machine.", resp.status_code, resp.text)
                    return self._process_fallback_turn(intent, entities, text, session)
        except Exception as e:
            logger.error("Gemini turn processing failed: %s. Falling back to local state machine.", e)
            return self._process_fallback_turn(intent, entities, text, session)

        session["history"].append({"role": "model", "text": response_text})
        
        return {
            "response": response_text,
            "state": session["state"],
            "tool_executed": tool_executed,
            "tool_result": tool_result
        }

    def _synthesize_final_reply(self, user_query: str, tool_name: str, tool_result: Dict[str, Any], api_key: str, language: str = "en") -> str:
        """Invokes Gemini to format tool output, with deterministic local fallback."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
        prompt = (
            f"The user query was: '{user_query}'\n"
            f"The backend tool '{tool_name}' was executed successfully.\n"
            f"Tool Execution Result: {json.dumps(tool_result)}\n\n"
            f"Based on this result, generate a concise, friendly, and natural voice response (1-2 sentences) "
            f"for the customer in '{language}'. Do not include JSON formatting."
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
        
        return self._format_tool_response(tool_name, tool_result, language=language)

    def _format_tool_response(self, tool_name: str, tool_result: Dict[str, Any], language: str = "en") -> str:
        """Generates a deterministic, language-consistent response string from backend tool results."""
        success = tool_result.get("success", False)
        is_hindi = (language or "en").lower() in ["hi", "hindi"]
        
        if tool_name == "get_order_status":
            if success:
                order_id = tool_result.get("order_id")
                status = tool_result.get("status")
                carrier = tool_result.get("carrier")
                date = tool_result.get("delivery_date")
                if is_hindi:
                    return f"आपका ऑर्डर {order_id} का स्टेटस '{status}' है। यह {carrier} द्वारा डिलीवर किया जा रहा है और अनुमानित तिथि {date} है।"
                return f"Order {order_id} is currently '{status}'. It is shipped via {carrier} and expected on {date}."
            else:
                if is_hindi:
                    return f"क्षमा करें, ऑर्डर आईडी {tool_result.get('order_id', '')} हमारे डेटाबेस में नहीं मिला।"
                return f"I'm sorry, order ID {tool_result.get('order_id', '')} was not found in our records."

        elif tool_name == "reset_password":
            if success:
                if is_hindi:
                    return "पासवर्ड रीसेट लिंक आपके रजिस्टर्ड ईमेल पर भेज दिया गया है।"
                return "A password reset token has been generated and sent to your email address."
            else:
                if is_hindi:
                    return "क्षमा करें, यह ईमेल हमारे डेटाबेस में नहीं मिला।"
                return "Sorry, that email address was not found in our database."

        elif tool_name == "update_address":
            if success:
                if is_hindi:
                    return "धन्यवाद। आपका डिलीवरी पता सफलतापूर्वक अपडेट कर दिया गया है।"
                return "Thank you. Your delivery address has been successfully updated."
            else:
                if is_hindi:
                    return "क्षमा करें, यह फोन नंबर हमारे रिकॉर्ड्स में नहीं मिला।"
                return "Sorry, that phone number was not found in our records."

        elif tool_name == "cancel_order":
            if success:
                if is_hindi:
                    return f"आपका ऑर्डर {tool_result.get('order_id', '')} सफलतापूर्वक रद्द (cancel) कर दिया गया है।"
                return f"Your order {tool_result.get('order_id', '')} has been cancelled successfully."
            else:
                if is_hindi:
                    return "क्षमा करें, ऑर्डर नहीं मिला।"
                return "Sorry, order was not found."

        elif tool_name == "refund_order":
            if success:
                res_oid = tool_result.get('order_id') or ""
                res_reason = tool_result.get('reason') or "damaged"
                if is_hindi:
                    return f"आपका रिफंड अनुरोध {res_oid} सफलतापूर्वक स्वीकार कर लिया गया है।"
                return f"I have processed a refund request for order {res_oid} because the item was {res_reason}."
            else:
                if is_hindi:
                    return "क्षमा करें, ऑर्डर नहीं मिला।"
                return "Sorry, order was not found."

        return tool_result.get("message", "Request processed successfully.")

    def _dispatch_tool(self, tool_name: str, entities: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dispatches tool execution to BackendClient."""
        ctx = context or {}
        logger.info("Executing tool '%s' with entities: %s, context: %s", tool_name, entities, ctx)
        
        if tool_name == "get_order_status":
            order_id = entities.get("order_id") or ctx.get("order_id") or ""
            return self.backend_client.get_order_status(order_id)
            
        elif tool_name == "reset_password":
            email = entities.get("email") or ctx.get("email") or ""
            return self.backend_client.reset_password(email)
            
        elif tool_name == "update_address":
            phone = entities.get("phone_number") or ctx.get("phone_number") or ""
            address = entities.get("address") or ctx.get("address") or ""
            return self.backend_client.update_address(phone, address)
            
        elif tool_name == "cancel_order":
            order_id = entities.get("order_id") or ctx.get("order_id") or ""
            return self.backend_client.cancel_order(order_id)
            
        elif tool_name == "refund_order":
            order_id = entities.get("order_id") or ctx.get("order_id") or ""
            reason = entities.get("reason") or ctx.get("reason") or "unspecified reason"
            return self.backend_client.refund_order(order_id, reason)
            
        else:
            logger.error("Unknown tool name: %s", tool_name)
            return {"success": False, "error": f"Unknown tool {tool_name}"}

    def _map_state_from_slots(self, missing_slots: List[str], intent: Optional[str]) -> DialogueState:
        """Maps missing slot lists into DialogueState enums."""
        if not missing_slots:
            if intent == "cancel_order":
                return DialogueState.AWAITING_CANCEL_CONFIRM
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
        is_hindi = (session.get("language") or "en").lower() in ["hi", "hindi"]
        
        for k, v in entities.items():
            if v is not None:
                context[k] = v

        if current_state != DialogueState.IDLE and intent not in ["unknown", context.get("workflow")]:
            self._reset_workflow_context(session)
            session["state"] = DialogueState.IDLE
            current_state = DialogueState.IDLE

        response_text = ""
        tool_executed = None
        tool_result = None

        if intent == "greeting":
            session["state"] = DialogueState.IDLE
            self._reset_workflow_context(session)
            response_text = "नमस्ते! वाणी कस्टमर सपोर्ट में आपका स्वागत है। मैं आपकी क्या मदद कर सकता हूँ?" if is_hindi else "Hello! Welcome to Vani Customer Support. How can I help you today? You can check order status, reset password, or update address."
            session["history"].append({"role": "model", "text": response_text})
            return {"response": response_text, "state": session["state"], "tool_executed": None, "tool_result": None}
            
        elif intent == "farewell":
            session["state"] = DialogueState.IDLE
            self._reset_workflow_context(session)
            response_text = "वाणी कस्टमर सपोर्ट चुनने के लिए धन्यवाद। आपका दिन शुभ हो! नमस्ते।" if is_hindi else "Thank you for choosing Vani Customer Support. Have a wonderful day ahead! Namaste."
            session["history"].append({"role": "model", "text": response_text})
            return {"response": response_text, "state": session["state"], "tool_executed": None, "tool_result": None}

        if current_state == DialogueState.IDLE:
            if intent == "order_status":
                context["workflow"] = "order_status"
                if "order_id" in context:
                    order_id = context["order_id"]
                    tool_executed = "get_order_status"
                    tool_result = self.backend_client.get_order_status(order_id)
                    response_text = self._format_tool_response(tool_executed, tool_result, language=session.get("language"))
                    session["state"] = DialogueState.IDLE
                    self._reset_workflow_context(session)
                else:
                    session["state"] = DialogueState.AWAITING_ORDER_ID
                    response_text = "कृपया अपना 6-अंकों का ऑर्डर ID बताएं।" if is_hindi else "I'd be happy to check your order status. Could you please state your 6-digit Order ID?"

            elif intent == "cancel_order":
                context["workflow"] = "cancel_order"
                if "order_id" in context:
                    order_id = context["order_id"]
                    session["state"] = DialogueState.AWAITING_CANCEL_CONFIRM
                    response_text = f"क्या आप निश्चित रूप से ऑर्डर {order_id} रद्द करना चाहते हैं? कृपया 'हाँ' या पुष्टि करें।" if is_hindi else f"Are you sure you want to cancel order {order_id}? Please say yes or confirm."
                else:
                    session["state"] = DialogueState.AWAITING_ORDER_ID
                    response_text = "मैं आपका ऑर्डर रद्द करने में मदद कर सकता हूँ। आपका 6-अंकों का ऑर्डर ID क्या है?" if is_hindi else "I can help you cancel your order. What is your 6-digit Order ID?"

            elif intent == "refund_request":
                context["workflow"] = "refund_request"
                if "order_id" not in context:
                    session["state"] = DialogueState.AWAITING_ORDER_ID
                    response_text = "हाँ, मैं रिफंड में आपकी मदद कर सकता हूँ। आपका 6-अंकों का ऑर्डर ID क्या है?" if is_hindi else "Sure, I can help you with a refund. What is your 6-digit Order ID?"
                elif "reason" not in context:
                    session["state"] = DialogueState.AWAITING_REASON
                    response_text = "कृपया रिफंड का कारण बताएं।" if is_hindi else "Could you please tell me the reason for the refund?"
                else:
                    order_id = context["order_id"]
                    reason = context["reason"]
                    tool_executed = "refund_order"
                    tool_result = self.backend_client.refund_order(order_id, reason)
                    response_text = self._format_tool_response(tool_executed, tool_result, language=session.get("language"))
                    session["state"] = DialogueState.IDLE
                    self._reset_workflow_context(session)

            elif intent == "password_reset":
                context["workflow"] = "password_reset"
                if "email" in context:
                    email = context["email"]
                    tool_executed = "reset_password"
                    tool_result = self.backend_client.reset_password(email)
                    response_text = self._format_tool_response(tool_executed, tool_result, language=session.get("language"))
                    session["state"] = DialogueState.IDLE
                    self._reset_workflow_context(session)
                else:
                    session["state"] = DialogueState.AWAITING_EMAIL
                    response_text = "पासवर्ड रीसेट करने के लिए कृपया अपना ईमेल पता दर्ज करें।" if is_hindi else "Sure, I can help you reset your password. What is your registered email address?"

            elif intent == "update_address":
                context["workflow"] = "update_address"
                if "phone_number" in context:
                    if "address" in context:
                        phone = context["phone_number"]
                        address = context["address"]
                        tool_executed = "update_address"
                        tool_result = self.backend_client.update_address(phone, address)
                        response_text = self._format_tool_response(tool_executed, tool_result, language=session.get("language"))
                        session["state"] = DialogueState.IDLE
                        self._reset_workflow_context(session)
                    else:
                        session["state"] = DialogueState.AWAITING_ADDRESS
                        response_text = "अपना नया डिलीवरी पता दर्ज करें।" if is_hindi else "I have your phone number. What is the new shipping address you would like to set?"
                else:
                    session["state"] = DialogueState.AWAITING_PHONE
                    response_text = "कृपया अपना 10-अंकों का फोन नंबर दर्ज करें।" if is_hindi else "I can help you update your address. First, could you tell me your 10-digit registered phone number?"
            else:
                response_text = "मैं आज आपकी और क्या मदद कर सकता हूँ?" if is_hindi else "How else can I help you today?"
                session["state"] = DialogueState.IDLE

        elif current_state == DialogueState.AWAITING_ORDER_ID:
            if "order_id" in context:
                order_id = context["order_id"]
                workflow = context.get("workflow", "order_status")
                if workflow == "cancel_order":
                    session["state"] = DialogueState.AWAITING_CANCEL_CONFIRM
                    response_text = f"क्या आप निश्चित रूप से ऑर्डर {order_id} रद्द करना चाहते हैं? कृपया 'हाँ' या पुष्टि करें।" if is_hindi else f"Are you sure you want to cancel order {order_id}? Please say yes or confirm."
                elif workflow == "refund_request":
                    if "reason" in context:
                        reason = context["reason"]
                        tool_executed = "refund_order"
                        tool_result = self.backend_client.refund_order(order_id, reason)
                        response_text = self._format_tool_response(tool_executed, tool_result, language=session.get("language"))
                        session["state"] = DialogueState.IDLE
                        self._reset_workflow_context(session)
                    else:
                        session["state"] = DialogueState.AWAITING_REASON
                        response_text = "कृपया रिफंड का कारण बताएं।" if is_hindi else "Could you please tell me the reason for the refund?"
                else:
                    tool_executed = "get_order_status"
                    tool_result = self.backend_client.get_order_status(order_id)
                    response_text = self._format_tool_response(tool_executed, tool_result, language=session.get("language"))
                    session["state"] = DialogueState.IDLE
                    self._reset_workflow_context(session)
            else:
                response_text = "कृपया अपना 6-अंकों का ऑर्डर ID दर्ज करें।" if is_hindi else "Please state your 6-digit Order ID."

        elif current_state == DialogueState.AWAITING_CANCEL_CONFIRM:
            clean_text = text.lower()
            if any(w in clean_text for w in ["yes", "confirm", "yeah", "haan", "ha", "sure", "हाँ", "हा", "कन्फर्म", "रद्द"]):
                order_id = context.get("order_id", "")
                tool_executed = "cancel_order"
                tool_result = self.backend_client.cancel_order(order_id)
                response_text = f"I have successfully cancelled your order {order_id}." if not is_hindi else f"आपका ऑर्डर {order_id} सफलतापूर्वक रद्द कर दिया गया है।"
                session["state"] = DialogueState.IDLE
                self._reset_workflow_context(session)
            else:
                response_text = "ऑर्डर कैंसलेशन रद्द कर दिया गया है।" if is_hindi else "Okay, I will not cancel your order. The cancellation request has been aborted."
                session["state"] = DialogueState.IDLE
                self._reset_workflow_context(session)

        elif current_state == DialogueState.AWAITING_EMAIL:
            if "email" in context:
                email = context["email"]
                tool_executed = "reset_password"
                tool_result = self.backend_client.reset_password(email)
                response_text = self._format_tool_response(tool_executed, tool_result, language=session.get("language"))
                session["state"] = DialogueState.IDLE
                self._reset_workflow_context(session)
            else:
                response_text = "कृपया अपना सही ईमेल पता प्रदान करें।" if is_hindi else "Please state a valid registered email address for your account."

        elif current_state == DialogueState.AWAITING_PHONE:
            if "phone_number" in context:
                session["state"] = DialogueState.AWAITING_ADDRESS
                response_text = "अपना नया डिलीवरी पता दर्ज करें।" if is_hindi else "Thank you. Now please state your new delivery address."
            else:
                response_text = "कृपया 10-अंकों का फोन नंबर प्रदान करें।" if is_hindi else "Please provide your 10-digit registered phone number."

        elif current_state == DialogueState.AWAITING_ADDRESS:
            if "address" in context:
                phone = context.get("phone_number", "")
                address = context["address"]
                tool_executed = "update_address"
                tool_result = self.backend_client.update_address(phone, address)
                response_text = self._format_tool_response(tool_executed, tool_result, language=session.get("language"))
                session["state"] = DialogueState.IDLE
                self._reset_workflow_context(session)
            else:
                response_text = "कृपया नया शिपिंग पता दर्ज करें।" if is_hindi else "Please state the new shipping address."

        elif current_state == DialogueState.AWAITING_REASON:
            if "reason" in context:
                order_id = context.get("order_id", "")
                reason = context["reason"]
                tool_executed = "refund_order"
                tool_result = self.backend_client.refund_order(order_id, reason)
                response_text = self._format_tool_response(tool_executed, tool_result, language=session.get("language"))
                session["state"] = DialogueState.IDLE
                self._reset_workflow_context(session)
            else:
                response_text = "कृपया रिफंड का कारण दर्ज करें।" if is_hindi else "Please state the reason for requesting a refund."

        session["history"].append({"role": "model", "text": response_text})
        
        return {
            "response": response_text,
            "state": session["state"],
            "tool_executed": tool_executed,
            "tool_result": tool_result
        }
