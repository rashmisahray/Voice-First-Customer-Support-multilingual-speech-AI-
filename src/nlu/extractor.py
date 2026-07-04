import abc
import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger("src.nlu.extractor")

class BaseEntityExtractor(abc.ABC):
    """Abstract base class for extracting entities from text."""

    @abc.abstractmethod
    def extract(self, text: str) -> Dict[str, Any]:
        """Extracts key entities (names, order IDs, phones, emails) from text.
        
        Args:
            text: Transcribed user speech text.
            
        Returns:
            Dict mapping entity types to extracted values.
        """
        pass

class MockEntityExtractor(BaseEntityExtractor):
    """Regex-based entity extractor simulating basic entity parsing."""
    
    def extract(self, text: str) -> Dict[str, Any]:
        logger.info("NLU Extractor: Extracting entities from text: %s", text)
        entities = {}

        # 1. Order ID Regex: e.g. ORD-123456 or ORD_876543
        order_match = re.search(r'\bORD[-_]?(\d+)\b', text, re.IGNORECASE)
        if order_match:
            entities["order_id"] = order_match.group(0).upper()

        # 2. Email Regex
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            entities["email"] = email_match.group(0)

        # 3. Phone number Regex: simplified 10-digit number
        phone_match = re.search(r'\b\d{10}\b', text)
        if phone_match:
            entities["phone_number"] = phone_match.group(0)

        logger.info("NLU Extractor: Extracted entities: %s", entities)
        return entities
