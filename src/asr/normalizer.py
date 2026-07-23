import re
import logging
from typing import Dict, Any

logger = logging.getLogger("src.asr.normalizer")

# Mapping of English and Hindi/Hinglish spoken digit words to digits
SPOKEN_DIGITS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "shunya": "0", "ek": "1", "do": "2", "teen": "3", "chaar": "4",
    "paanch": "5", "chhah": "6", "saat": "7", "aath": "8", "nau": "9"
}

class TranscriptNormalizer:
    """
    Normalizer module that post-processes raw transcripts from ASR,
    standardizing spoken email addresses and spoken number sequences
    into clean, normalized representations for downstream NLU.
    """

    def normalize(self, text: str) -> str:
        """
        Applies email normalization, spoken number word translation,
        and digit sequence collapsing to the input text.
        """
        if not text or text.strip() == "":
            return text

        logger.info("Normalizing raw transcript: '%s'", text)
        normalized = text

        # 1. Normalize spoken emails
        normalized = self._normalize_emails(normalized)

        # 2. Translate spoken digit words (e.g. "one", "two", "ek", "do") to digits
        normalized = self._translate_digit_words(normalized)

        # 3. Collapse consecutive digits separated by spaces, commas, or hyphens
        normalized = self._collapse_consecutive_digits(normalized)

        logger.info("Normalized transcript result: '%s'", normalized)
        return normalized

    def _normalize_emails(self, text: str) -> str:
        """
        Normalizes spoken email components like "at", "at the rate", "dot", "underscore", "dash"
        into proper email syntax (e.g., john@example.com).
        """
        # 1. Pre-normalize connectives (underscore, dash, hyphen, dot) globally when likely part of an email candidate
        processed = re.sub(r'\s+(?:underscore)\s+', '_', text, flags=re.IGNORECASE)
        processed = re.sub(r'\s+(?:dash|hyphen|minus)\s+', '-', processed, flags=re.IGNORECASE)
        processed = re.sub(r'\s+(?:dot|\.|\s+dot\s+)\s*', '.', processed, flags=re.IGNORECASE)
        
        # 2. Match email candidates: username + at/at the rate + domain parts (e.g. example.com)
        email_pattern = re.compile(
            r'\b([\w_.-]+)\s+(?:at\s+the\s+rate\s+of|at\s+the\s+rate|at\s*rate|atrate|adurate|at)\s+([\w_.-]+\.[\w_.-]+)\b',
            re.IGNORECASE
        )

        def replacer(match):
            user = match.group(1)
            domain = match.group(2)

            # Remove spaces if any
            user = re.sub(r'\s+', '', user)
            domain = re.sub(r'\s+', '', domain)
            
            return f"{user.lower()}@{domain.lower()}"

        return email_pattern.sub(replacer, processed)

    def _translate_digit_words(self, text: str) -> str:
        """
        Replaces individual spoken digit words (English/Hindi) with their corresponding digits.
        Uses word boundaries to avoid replacing parts of larger words.
        """
        result = text
        for word, digit in SPOKEN_DIGITS.items():
            pattern = re.compile(rf'\b{word}\b', re.IGNORECASE)
            result = pattern.sub(digit, result)
        return result

    def _collapse_consecutive_digits(self, text: str) -> str:
        """
        Finds sequences of digits separated only by spaces, commas, or hyphens
        and collapses them into a single continuous numeric string.
        """
        # Matches sequences of digits separated only by spaces, commas, or hyphens
        digit_seq_pattern = re.compile(r'\b\d(?:[\s,-]+\d)+\b')

        def replacer(match):
            # Remove all spaces, commas, and hyphens in the matched sequence
            return re.sub(r'[\s,-]', '', match.group(0))

        return digit_seq_pattern.sub(replacer, text)
