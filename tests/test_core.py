import logging
from src.core.config import settings, load_settings
from src.core.logger import PIIMaskingFilter

def test_settings_load():
    """Verify that settings are parsed correctly with expected default structures."""
    assert settings.app.name == "Vani Voice AI Support"
    assert settings.logging.level in ["INFO", "DEBUG", "WARNING", "ERROR"]
    
    # Load specific test paths
    loaded = load_settings()
    assert loaded.server.port == 8000

def test_pii_masking_filter():
    """Test that PIIMaskingFilter successfully masks emails, phone numbers, and order numbers."""
    mask_filter = PIIMaskingFilter(mask_char="*", enabled=True)
    
    class MockRecord(logging.LogRecord):
        def __init__(self, msg, args=()):
            super().__init__(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=msg,
                args=args,
                exc_info=None
            )

    # 1. Test Email Masking
    record = MockRecord("User email is alex.jones@domain.co.in, please reset.")
    mask_filter.filter(record)
    assert "@domain.co.in" in record.msg
    assert "a********s@domain.co.in" in record.msg
    assert "alex.jones" not in record.msg

    # 2. Test Phone Number Masking
    record = MockRecord("Call back customer at +91 9876543210 immediately.")
    mask_filter.filter(record)
    assert "+91 9876543210" not in record.msg
    assert "Phone(" in record.msg

    # 3. Test Order ID Masking
    record = MockRecord("Looking up order ORD-12345678 in database.")
    mask_filter.filter(record)
    assert "ORD-12345678" not in record.msg
    assert "ORD-****5678" in record.msg

    # 4. Test Dict structured logging masking
    record_dict = MockRecord({"user_email": "hello@test.com", "msg": "Phone is 9998887777"})
    mask_filter.filter(record_dict)
    assert record_dict.msg["user_email"] == "h***o@test.com"
    assert "9998887777" not in record_dict.msg["msg"]
