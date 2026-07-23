import logging
from typing import Dict, Any
from src.core.config import settings

logger = logging.getLogger("src.nlu.classifier")

class BaseIntentClassifier:
    """Base class for Intent Classification."""
    def classify(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

class MockIntentClassifier(BaseIntentClassifier):
    """
    Modular keyword and rule-based Intent Classifier for Vani.
    Supports 20 production customer support intents in English, Hindi, and Hinglish.
    """

    def __init__(self):
        # Define trigger keyword lists for 20 intents
        self.rules = {
            "greeting": [
                "hello", "hi", "namaste", "hey", "good morning", "good evening", "greetings", "kaise", "kya haal"
            ],
            "farewell": [
                "bye", "goodbye", "exit", "quit", "thank you", "thanks", "see you", "alvida", "dhanyawad", "shukriya"
            ],
            "order_status": [
                "order status", "track order", "where is my order", "order id", "check order", "my package", 
                "delivery status", "order", "status", "track", "package", "delivery", "kahan hai", "kab aayega"
            ],
            "password_reset": [
                "reset password", "change password", "forgot password", "password reset", "reset my pass", 
                "account access", "password", "reset", "passwal", "panswai", "karthu", "pass", "login", "badal"
            ],
            "update_address": [
                "update address", "change address", "new address", "shipping address", "delivery address", 
                "update my address", "address", "location", "pata", "ptta", "badalna", "update", "makan"
            ],
            "cancel_order": [
                "cancel order", "cancel my order", "abort order", "cancel karo", "order cancel", "stop order"
            ],
            "refund_request": [
                "refund", "money back", "refund request", "paise wapas", "refund chahiye", "return money"
            ],
            "payment_issue": [
                "payment failed", "payment issue", "declined", "card declined", "failed transaction", "payment", 
                "transaction error", "paise kat gaye", "billing", "charges"
            ],
            "product_inquiry": [
                "in stock", "product information", "size chart", "is it available", "availability", "specifications", 
                "product inquiry", "details about", "stock", "color options"
            ],
            "shipping_policy": [
                "shipping time", "how long to ship", "shipping cost", "delivery fee", "delivery policy", 
                "international shipping", "shipping policy", "delivers to"
            ],
            "return_policy": [
                "return policy", "how to return", "return window", "return label", "can i return", 
                "wapas karna", "return process"
            ],
            "human_agent": [
                "talk to human", "live agent", "customer support representative", "speak to someone", "human", 
                "agent", "representative", "operator", "call support", "live support"
            ],
            "store_hours": [
                "store hours", "what time do you open", "closing time", "open hours", "timings", "when does store close"
            ],
            "store_location": [
                "store address", "where is the store", "directions to store", "nearest store", "find store", "location"
            ],
            "account_creation": [
                "create account", "sign up", "register", "join", "new user", "make an account"
            ],
            "promotions_coupons": [
                "coupon", "discount code", "promo", "discount", "offer", "sale", "deals", "promotions"
            ],
            "feedback_complaint": [
                "complaint", "feedback", "file a complaint", "bad service", "review", "suggest", "unhappy"
            ],
            "order_modification": [
                "modify order", "change items", "change order", "add items", "remove items", "order edit"
            ],
            "damaged_item": [
                "damaged", "broken", "defective", "torn", "not working", "damaged item", "received broken"
            ],
            "warranty_info": [
                "warranty", "guarantee", "warranty period", "is it covered", "warranty policy"
            ]
        }

    def classify(self, text: str) -> Dict[str, Any]:
        logger.info("NLU Classifier: Processing text: '%s'", text)
        
        text_lower = text.lower().strip()
        
        # Calculate matching scores based on keyword occurrences
        scores = {}
        for intent, keywords in self.rules.items():
            matches = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Grant higher weight for multi-word phrase matches
                    matches += 1.5 if len(keyword.split()) > 1 else 1.0
            
            if matches > 0:
                # Calculate normalized confidence score
                scores[intent] = min(0.99, 0.70 + (0.1 * matches))
        
        # Find the highest scoring intent
        if scores:
            best_intent = max(scores, key=scores.get)
            best_score = scores[best_intent]
        else:
            best_intent = "unknown"
            best_score = 0.0
            
        # Fallback if confidence is below threshold config
        threshold = settings.nlu.intent_confidence_threshold
        if best_score < threshold:
            logger.warning(
                "NLU Classifier: Top intent '%s' (score: %.2f) below threshold %.2f. Falling back to unknown.",
                best_intent, best_score, threshold
            )
            best_intent = "unknown"
            best_score = 0.0
            
        result = {
            "intent": best_intent,
            "confidence": round(best_score, 2)
        }
        
        logger.info("NLU Classifier Result: Intent: '%s', Confidence: %.2f", best_intent, best_score)
        return result
