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
    Analyzes user transcript and returns classified intent and confidence score.
    """

    def __init__(self):
        # Define trigger keyword lists for each target intent
        self.rules = {
            "greeting": ["hello", "hi", "namaste", "hey", "good morning", "good evening", "greetings"],
            "farewell": ["bye", "goodbye", "exit", "quit", "thank you", "thanks", "see you", "alvida"],
            "order_status": ["order status", "track order", "where is my order", "order id", "check order", "my package", "delivery status"],
            "password_reset": ["reset password", "change password", "forgot password", "password reset", "reset my pass", "account access"],
            "update_address": ["update address", "change address", "new address", "shipping address", "delivery address", "update my address"]
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
                    # Grant higher weight for exact phrase matches
                    matches += 1.5 if len(keyword.split()) > 1 else 1.0
            
            if matches > 0:
                # Calculate simple normalized score
                scores[intent] = min(0.99, 0.7 + (0.1 * matches))
        
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
