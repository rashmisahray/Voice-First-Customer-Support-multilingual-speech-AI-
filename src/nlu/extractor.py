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
    """

    def extract(self, text: str) -> Dict[str, Any]:
        logger.info("NLU Extractor: Extracting entities from text: '%s'", text)
        entities = {}

        # 1. Order ID: Matches ORD-123456, ORD123456, or stand-alone 6-digit integers
        order_match = re.search(r'\b(?:ORD[-_]?)?(\d{6})\b', text, re.IGNORECASE)
        if order_match:
            # Normalize to standardized format e.g. ORD-123456
            digits = order_match.group(1)
            entities["order_id"] = f"ORD-{digits}"
            logger.info("NLU Extractor: Extracted Order ID: %s", entities["order_id"])

        # 2. Phone Number: Matches standard 10-digit formats (e.g. 9876543210, +91-9876543210, +91 98765 43210)
        phone_match = re.search(r'\b(?:\+?91[-.\s]?)?([6-9]\d{9})\b', text)
        if phone_match:
            entities["phone_number"] = phone_match.group(1)
            logger.info("NLU Extractor: Extracted Phone Number: %s", entities["phone_number"])

        # 3. Email: Standard email pattern
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            entities["email"] = email_match.group(0)
            logger.info("NLU Extractor: Extracted Email: %s", entities["email"])

        # 4. Address: If user specifies an address (e.g. "address is 123 Main Street" or "ship to 123 Main St")
        address_patterns = [
            r'(?:address is|ship to|deliver to|address to|living at)\s+(.+)$',
            r'(?:new address is|updated address is)\s+(.+)$'
        ]
        for pattern in address_patterns:
            address_match = re.search(pattern, text, re.IGNORECASE)
            if address_match:
                # Strip punctuation and trailing details
                addr = address_match.group(1).strip()
                if addr:
                    entities["address"] = addr
                    logger.info("NLU Extractor: Extracted Address: %s", entities["address"])
                    break

        return entities
