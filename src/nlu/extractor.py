import os
import logging
import re
import httpx
import json
from typing import Dict, Any

logger = logging.getLogger("src.nlu.extractor")

# List of keywords for local offline fallback extraction
PRODUCT_KEYWORDS = ["shirt", "tshirt", "t-shirt", "shoes", "jeans", "jacket", "laptop", "phone", "watch", "mobile", "charger"]
DATE_TIME_KEYWORDS = ["tomorrow", "today", "yesterday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "morning", "afternoon", "evening", "kal", "aaj"]
REASON_KEYWORDS = ["damaged", "broken", "torn", "wrong size", "late delivery", "changed my mind", "defective", "kharab", "tuta", "mistake"]
PAYMENT_KEYWORDS = ["credit card", "debit card", "upi", "net banking", "cash on delivery", "card", "gpay", "phonepe", "paypal", "paytm", "cash"]

class BaseEntityExtractor:
    """Base class for Entity Extraction."""
    def extract(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

class LLMEntityExtractor(BaseEntityExtractor):
    """
    LLM-based Entity Extractor for Vani.
    Extracts 9 entity types (order_id, phone_number, email, address, customer_name,
    product_name, date_time, reason, payment_method) using Gemini/OpenAI API, 
    with a robust offline regex/rules fallback when API keys are missing.
    """

    def extract(self, text: str) -> Dict[str, Any]:
        logger.info("LLM Extractor: Extracting entities from text: '%s'", text)
        
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        if gemini_key:
            logger.info("LLM Extractor: Calling live Gemini API for entity extraction...")
            try:
                entities = self._call_gemini_extraction(text, gemini_key)
                if entities:
                    return entities
            except Exception as e:
                logger.error("Gemini Entity extraction failed: %s. Falling back...", e)

        elif openai_key:
            logger.info("LLM Extractor: Calling live OpenAI API for entity extraction...")
            try:
                entities = self._call_openai_extraction(text, openai_key)
                if entities:
                    return entities
            except Exception as e:
                logger.error("OpenAI Entity extraction failed: %s. Falling back...", e)

        # Fallback to local rules-based parser
        logger.info("LLM Extractor: Using local rules-based entity extractor fallback...")
        return self._extract_local_fallback(text)

    def _call_gemini_extraction(self, text: str, api_key: str) -> Dict[str, Any] | None:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        prompt = (
            "You are an NLP entity extraction system. Extract entities from the user query and return "
            "ONLY a valid JSON object. Do not include markdown code block formatting. "
            "Extract these fields: order_id (normalize to ORD-XXXXXX), phone_number (10 digits), email, "
            "address, customer_name, product_name, date_time, reason, payment_method. "
            "If an entity is not found, set its value to null.\n"
            f"User query: '{text}'\n"
            "JSON Response:"
        )
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1
            }
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Strip markdown blocks if returned
                content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE)
                return json.loads(content)
        return None

    def _call_openai_extraction(self, text: str, api_key: str) -> Dict[str, Any] | None:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        prompt = (
            "Extract entities from the user query and return ONLY a valid JSON object. "
            "Fields: order_id (normalize to ORD-XXXXXX), phone_number (10 digits), email, "
            "address, customer_name, product_name, date_time, reason, payment_method. "
            "If an entity is not found, set its value to null."
        )
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                return json.loads(content)
        return None

    def _extract_local_fallback(self, text: str) -> Dict[str, Any]:
        """
        Offline rule-based entity extractor fallback matching regexes and keywords.
        """
        entities = {}

        # Pre-process text to remove commas from formatted numbers e.g. "876,543" -> "876543"
        cleaned_text = re.sub(r'(\d),(\d)', r'\1\2', text)

        # 1. Order ID: Matches ORD-123456, ORD123456, or stand-alone 6-digit integers
        order_match = re.search(r'\b(?:ORD[-_]?)?(\d{6})\b', cleaned_text, re.IGNORECASE)
        if order_match:
            digits = order_match.group(1)
            entities["order_id"] = f"ORD-{digits}"

        # 2. Phone Number: Matches standard 10-digit formats (e.g. 9876543210, +91-9876543210)
        phone_match = re.search(r'\b(?:\+?91[-.\s]?)?([6-9]\d{9})\b', cleaned_text)
        if phone_match:
            entities["phone_number"] = phone_match.group(1)

        # 3. Email: Standard email pattern & spoken ASR phonetic variations
        spoken_text = re.sub(r'at\s*the\s*rate,?\s*|adurate|ad\s*rate|at\s*rate|atrate|drate|\bat\b', '@', cleaned_text, flags=re.IGNORECASE)
        spoken_text = re.sub(r'\s*@,?\s*', '@', spoken_text)
        spoken_text = re.sub(r'\s+dot\s+', '.', spoken_text, flags=re.IGNORECASE)

        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', spoken_text)
        if email_match:
            entities["email"] = email_match.group(0).lower()

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
                    break

        # 5. Customer Name: Matches "my name is X" / "call me X" / "mera naam X hai"
        name_match = re.search(r'\b(?:my name is|call me|name is|naam hai|naam)\s+([A-Za-z]+)\b', cleaned_text, re.IGNORECASE)
        if name_match:
            entities["customer_name"] = name_match.group(1).capitalize()

        # 6. Product Name keyword matching
        for k in PRODUCT_KEYWORDS:
            if re.search(rf'\b{k}\b', cleaned_text, re.IGNORECASE):
                entities["product_name"] = k
                break

        # 7. Date Time keyword matching
        for k in DATE_TIME_KEYWORDS:
            if re.search(rf'\b{k}\b', cleaned_text, re.IGNORECASE):
                entities["date_time"] = k
                break

        # 8. Reason keyword matching
        for k in REASON_KEYWORDS:
            if re.search(rf'\b{k}\b', cleaned_text, re.IGNORECASE):
                entities["reason"] = k
                break

        # 9. Payment Method keyword matching
        for k in PAYMENT_KEYWORDS:
            if re.search(rf'\b{k}\b', cleaned_text, re.IGNORECASE):
                entities["payment_method"] = k
                break

        # Normalize empty values to None
        for k in ["order_id", "phone_number", "email", "address", "customer_name", "product_name", "date_time", "reason", "payment_method"]:
            if k not in entities:
                entities[k] = None

        return entities

# Backward compatibility alias
MockEntityExtractor = LLMEntityExtractor
