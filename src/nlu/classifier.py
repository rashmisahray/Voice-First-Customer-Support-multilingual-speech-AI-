import abc
import logging
from typing import Dict, Any
from src.core.config import settings

logger = logging.getLogger("src.nlu.classifier")

# Predefined list of sample intents from our future 50+ list
SAMPLE_INTENTS = [
    "order_status", "cancel_order", "return_product", "refund_request",
    "update_address", "password_reset", "login_issue", "delivery_delay",
    "agent_transfer", "greeting", "farewell", "payment_issue"
]

class BaseIntentClassifier(abc.ABC):
    """Abstract base class for classifying user intents from text."""

    @abc.abstractmethod
    def classify(self, text: str) -> Dict[str, Any]:
        """Classifies the user intent from transcribed text.
        
        Args:
            text: Transcribed user speech text.
            
        Returns:
            Dict containing the identified intent and confidence score.
        """
        pass

class MockIntentClassifier(BaseIntentClassifier):
    """Mock intent classifier that matches predefined keywords to intents."""
    
    def classify(self, text: str) -> Dict[str, Any]:
        logger.info("NLU Classifier: Processing text: %s", text)
        
        text_lower = text.lower()
        intent = "unknown"
        confidence = 0.0

        if "order" in text_lower or "status" in text_lower:
            intent = "order_status"
            confidence = 0.95
        elif "cancel" in text_lower:
            intent = "cancel_order"
            confidence = 0.92
        elif "refund" in text_lower:
            intent = "refund_request"
            confidence = 0.89
        elif "address" in text_lower or "pata" in text_lower:
            intent = "update_address"
            confidence = 0.90
        elif "password" in text_lower or "reset" in text_lower:
            intent = "password_reset"
            confidence = 0.94
        elif "hello" in text_lower or "hi" in text_lower or "namaste" in text_lower:
            intent = "greeting"
            confidence = 0.99
            
        # Enforce confidence threshold configured in configs/config.yaml
        if confidence < settings.nlu.intent_confidence_threshold:
            logger.warning(
                "NLU Classifier: Intent %s confidence %f below threshold %f. Falling back.",
                intent, confidence, settings.nlu.intent_confidence_threshold
            )
            intent = "unknown"
            confidence = 0.0
            
        result = {"intent": intent, "confidence": confidence}
        logger.info("NLU Classifier: Classified intent: %s (confidence: %.2f)", intent, confidence)
        return result
