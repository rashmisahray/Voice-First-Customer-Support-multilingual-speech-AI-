import os
import logging
import httpx
from src.nlu.classifier import MockIntentClassifier

logger = logging.getLogger("src.nlu.llm_fallback")

# Predefined contextual templates for offline fallback across all 20 intents
FALLBACK_RESPONSES = {
    "greeting": "Hello! Welcome to Vani Customer Support. How can I help you today?",
    "farewell": "Thank you for contacting Vani Customer Support. Have a wonderful day ahead! Namaste.",
    "order_status": "I can help you check your order status. Could you please state your 6-digit Order ID?",
    "password_reset": "I can help you reset your password. Please provide your registered email address.",
    "update_address": "I can help you update your delivery address. First, could you tell me your 10-digit registered phone number?",
    "cancel_order": "I can help you cancel an active order. Please provide your 6-digit Order ID to proceed.",
    "refund_request": "To request a refund, please share your 6-digit Order ID and the reason for the refund.",
    "payment_issue": "If your payment transaction failed, please check your card status or try another payment method.",
    "product_inquiry": "For product specifications, sizes, or stock availability, please check the product details page on our website.",
    "shipping_policy": "Standard shipping takes 3 to 5 business days. Express shipping options take 1 to 2 business days.",
    "return_policy": "You can return any item within 30 days of purchase. Return shipping labels are free and printable online.",
    "human_agent": "Sure, connecting you to a live customer support representative now. Please hold for a moment.",
    "store_hours": "Our physical store is open Monday through Saturday, from 9:00 AM to 8:00 PM. We are closed on Sundays.",
    "store_location": "Our flagship retail store is located at 124 MG Road, Bangalore.",
    "account_creation": "To create a new account, please click the 'Register' button at the top right of our web homepage.",
    "promotions_coupons": "You can use the coupon code 'WELCOME10' to get an instant 10% discount on your first order!",
    "feedback_complaint": "Thank you for your feedback. We have recorded your concern, and a manager will review it within 24 hours.",
    "order_modification": "To modify items in your order, please call us directly or cancel and place a new order.",
    "damaged_item": "I am sorry you received a damaged item. Please email a photo of the item to support@example.com for a replacement.",
    "warranty_info": "All our electronic items come with a standard 1-year limited manufacturer warranty covering defects."
}

class LLMFallback:
    """
    LLM Fallback engine that routes unknown or low-confidence queries to an LLM
    (Gemini or OpenAI if API keys are set), or falls back to an intelligent local mock
    responder matching user query topics to 20 customer support intents.
    """

    def __init__(self):
        self.classifier = MockIntentClassifier()

    def generate_response(self, text: str, session_id: str = "default") -> str:
        """
        Generates a concise customer support response using live APIs or local mock fallback.
        """
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        if gemini_key:
            logger.info("LLM Fallback: Calling live Gemini API...")
            try:
                response = self._call_gemini(text, gemini_key)
                if response:
                    return response
            except Exception as e:
                logger.error("Gemini API call failed: %s. Falling back...", e)

        elif openai_key:
            logger.info("LLM Fallback: Calling live OpenAI API...")
            try:
                response = self._call_openai(text, openai_key)
                if response:
                    return response
            except Exception as e:
                logger.error("OpenAI API call failed: %s. Falling back...", e)

        # Local smart fallback
        logger.info("LLM Fallback: Using local smart fallback responder...")
        return self._generate_local_fallback(text)

    def _call_gemini(self, text: str, api_key: str) -> str | None:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        prompt = (
            "You are Vani, a helpful multilingual customer support voice assistant. "
            "Answer the user's question concisely in the language they used (English, Hindi, or Hinglish). "
            f"Keep your answer under 2 sentences. User query: '{text}'"
        )
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": 100,
                "temperature": 0.5
            }
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return None

    def _call_openai(self, text: str, api_key: str) -> str | None:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are Vani, a helpful multilingual customer support voice assistant. Answer concisely in under 2 sentences in the user's language (English, Hindi, or Hinglish)."},
                {"role": "user", "content": text}
            ],
            "max_tokens": 100,
            "temperature": 0.5
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        return None

    def _generate_local_fallback(self, text: str) -> str:
        """
        Offline rule-based fallback that matches keywords to one of the 20 customer support intents
        to provide a highly realistic answer.
        """
        # Call classifier to see if it matches any of the 20 intents under a lower threshold
        classification = self.classifier.classify(text)
        intent = classification["intent"]
        
        # If classifier yielded a valid intent, return its predefined template
        if intent != "unknown" and intent in FALLBACK_RESPONSES:
            return FALLBACK_RESPONSES[intent]
            
        # General fallback if completely off-topic or unknown
        return "I'm here to help you. Could you please specify if you'd like to check order status, reset password, or update address?"
