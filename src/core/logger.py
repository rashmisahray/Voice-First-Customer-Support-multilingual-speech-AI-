import logging
import os
import re
from pathlib import Path
from src.core.config import settings

# Regular expressions for common PII patterns
EMAIL_REGEX = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)
# Matches 10-digit phone numbers and international formats (e.g., +91 98765 43210)
PHONE_REGEX = re.compile(
    r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
)
# Matches typical order numbers (e.g., ORD-12345678, Order: 98765432)
ORDER_ID_REGEX = re.compile(
    r'\b(?:ORD|ORDER)[-_\s]?#?\s*(\d{4,12})\b', re.IGNORECASE
)

class PIIMaskingFilter(logging.Filter):
    """
    Logging filter that intercepts log messages and masks sensitive information:
    - Email Addresses
    - Phone Numbers
    - Order IDs
    """
    def __init__(self, mask_char: str = "*", enabled: bool = True):
        super().__init__()
        self.mask_char = mask_char
        self.enabled = enabled

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.enabled:
            return True

        if isinstance(record.msg, str):
            record.msg = self.mask_text(record.msg)
        elif isinstance(record.msg, dict):
            # If logging a dictionary/structured log, mask string values
            record.msg = self.mask_dict(record.msg)

        # Also mask values in args if they are strings
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    new_args.append(self.mask_text(arg))
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)

        return True

    def mask_text(self, text: str) -> str:
        # 1. Mask Emails
        def email_repl(match):
            email = match.group(0)
            parts = email.split('@')
            if len(parts) == 2:
                username, domain = parts
                if len(username) > 2:
                    masked_username = username[0] + (self.mask_char * (len(username) - 2)) + username[-1]
                else:
                    masked_username = self.mask_char * len(username)
                return f"{masked_username}@{domain}"
            return self.mask_char * len(email)

        text = EMAIL_REGEX.sub(email_repl, text)

        # 2. Mask Phone Numbers
        def phone_repl(match):
            phone = match.group(0)
            # Remove digits keeping spaces/separators, or replace digits
            # Let's keep first 3 and last 2 characters (e.g. +91 and final digits) or mask the inner ones
            clean_digits = re.sub(r'\D', '', phone)
            if len(clean_digits) >= 7:
                # Mask all but first 2 and last 2 digits
                masked = clean_digits[0:2] + (self.mask_char * (len(clean_digits) - 4)) + clean_digits[-2:]
                # Try to map back to original spacing/symbols if needed, or return a simpler masked representation
                return f"Phone({masked})"
            return self.mask_char * len(phone)

        text = PHONE_REGEX.sub(phone_repl, text)

        # 3. Mask Order IDs
        def order_repl(match):
            order_full = match.group(0)
            digits = match.group(1)
            # Mask inner digits of order number
            if len(digits) > 4:
                masked_digits = (self.mask_char * (len(digits) - 4)) + digits[-4:]
            else:
                masked_digits = self.mask_char * len(digits)
            # Reconstruct the order ID match
            return order_full.replace(digits, masked_digits)

        text = ORDER_ID_REGEX.sub(order_repl, text)

        return text

    def mask_dict(self, d: dict) -> dict:
        new_dict = {}
        for k, v in d.items():
            if isinstance(v, str):
                new_dict[k] = self.mask_text(v)
            elif isinstance(v, dict):
                new_dict[k] = self.mask_dict(v)
            else:
                new_dict[k] = v
        return new_dict


def setup_logging():
    """Configures system-wide logging with PII masking."""
    log_level_str = settings.logging.level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers = []

    # Create formatters
    log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    formatter = logging.Formatter(log_format)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File Handler (if configured)
    if settings.logging.log_to_file:
        log_file = Path(settings.logging.log_file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    # Instantiate and add PII masking filter to all handlers
    pii_filter = PIIMaskingFilter(
        mask_char=settings.logging.pii_mask_char,
        enabled=settings.logging.mask_pii
    )
    
    for handler in root_logger.handlers:
        handler.addFilter(pii_filter)

    # Initialize a test statement
    logger = logging.getLogger("src.core.logger")
    logger.info("Logging initialized with PII masking: %s", settings.logging.mask_pii)
