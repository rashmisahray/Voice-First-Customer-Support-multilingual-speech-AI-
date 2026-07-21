import logging
import re
from typing import Dict, Any

logger = logging.getLogger("src.nlu.extractor")

class BaseEntityExtractor:
    """Base class for Entity Extraction."""
    def extract(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

class MockEntityExtractor(BaseEntityExtractor):
    """
    Regex-based Entity Extractor for Vani.
    Extracts Order IDs, phone numbers, email addresses, and address strings.
    Handles formatting (commas, spaces) and ASR phonetic spoken variations (e.g. adurate -> @).
    """

    def extract(self, text: str) -> Dict[str, Any]:
        logger.info("NLU Extractor: Extracting entities from text: '%s'", text)
        entities = {}

        # Pre-process text to remove commas from formatted numbers e.g. "876,543" -> "876543"
        cleaned_text = re.sub(r'(\d),(\d)', r'\1\2', text)

        # 1. Order ID: Matches ORD-123456, ORD123456, or stand-alone 6-digit integers
        order_match = re.search(r'\b(?:ORD[-_]?)?(\d{6})\b', cleaned_text, re.IGNORECASE)
        if order_match:
            digits = order_match.group(1)
            entities["order_id"] = f"ORD-{digits}"
            logger.info("NLU Extractor: Extracted Order ID: %s", entities["order_id"])

        # 2. Phone Number: Matches standard 10-digit formats (e.g. 9876543210, +91-9876543210)
        phone_match = re.search(r'\b(?:\+?91[-.\s]?)?([6-9]\d{9})\b', cleaned_text)
        if phone_match:
            entities["phone_number"] = phone_match.group(1)
            logger.info("NLU Extractor: Extracted Phone Number: %s", entities["phone_number"])

        # 3. Email: Standard email pattern & spoken ASR phonetic variations
        # Normalize spoken '@' pronunciations: "adurate", "ad rate", "at rate", "atrate", "drate", "at"
        spoken_text = re.sub(r'adurate|ad\s*rate|at\s*rate|atrate|drate|\bat\b', '@', cleaned_text, flags=re.IGNORECASE)
        spoken_text = re.sub(r'\s*@\s*', '@', spoken_text)
        spoken_text = re.sub(r'\s+dot\s+', '.', spoken_text, flags=re.IGNORECASE)

        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', spoken_text)
        if email_match:
            entities["email"] = email_match.group(0).lower()
            logger.info("NLU Extractor: Extracted Email: %s", entities["email"])

        # 4. Address: Match common address prefix patterns
        address_patterns = [
            r'(?:address is|ship to|deliver to|address to|living at|my address is)\s+(.+)$',
            r'(?:new address is|updated address is)\s+(.+)$'
        ]
        for pattern in address_patterns:
            address_match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if address_match:
                addr = address_match.group(1).strip()
                if addr:
                    entities["address"] = addr
                    logger.info("NLU Extractor: Extracted Address: %s", entities["address"])
                    break

        return entities
