import pytest
from src.asr.normalizer import TranscriptNormalizer

@pytest.fixture
def normalizer():
    return TranscriptNormalizer()

def test_email_normalization(normalizer):
    # Test basic spoken email with dot
    assert normalizer.normalize("My email is john at example dot com") == "My email is john@example.com"
    
    # Test at the rate
    assert normalizer.normalize("john at the rate example dot com") == "john@example.com"
    
    # Test at the rate of
    assert normalizer.normalize("john at the rate of example dot com") == "john@example.com"
    
    # Test underscore normalization
    assert normalizer.normalize("john underscore smith at example dot com") == "john_smith@example.com"
    
    # Test dash and hyphen normalization
    assert normalizer.normalize("john dash smith hyphen test at example dot com") == "john-smith-test@example.com"
    
    # Test capitalization
    assert normalizer.normalize("JOHN AT THE RATE EXAMPLE DOT COM") == "john@example.com"

def test_digit_collapsing(normalizer):
    # Test consecutive English digit words
    assert normalizer.normalize("eight seven six five four three") == "876543"
    
    # Test consecutive digits with spaces
    assert normalizer.normalize("8 7 6 5 4 3") == "876543"
    
    # Test consecutive digits with commas
    assert normalizer.normalize("8, 7, 6, 5, 4, 3") == "876543"
    
    # Test consecutive digits with hyphens
    assert normalizer.normalize("8-7-6-5-4-3") == "876543"
    
    # Test mixed text with digit sequence
    assert normalizer.normalize("My order ID is 8, 7, 6, 5, 4, 3") == "My order ID is 876543"
    
    # Test mixed digit words and digits
    assert normalizer.normalize("My order ID is eight 7 six 5 four 3") == "My order ID is 876543"
    
    # Test phone number collapsing
    assert normalizer.normalize("9 8 7 6 5 4 3 2 1 0") == "9876543210"

def test_mixed_hindi_english_speech(normalizer):
    # Test Hindi digit words collapsing
    assert normalizer.normalize("Mera order ID hai aath saat chhah paanch chaar teen") == "Mera order ID hai 876543"
    
    # Test Hinglish context
    assert normalizer.normalize("Namaste Mera order ID do chaar paanch hai") == "Namaste Mera order ID 245 hai"
    
    # Test mixed Hindi and English digit words
    assert normalizer.normalize("Mera order ID hai eight saat six paanch four teen") == "Mera order ID hai 876543"

def test_punctuation_and_conversational_text(normalizer):
    # Test punctuation preserve at the end of email
    assert normalizer.normalize("Please mail to john at example dot com.") == "Please mail to john@example.com."
    
    # Test punctuation preserve at the end of digit sequence
    assert normalizer.normalize("My phone: 9 8 7 6 5 4 3 2 1 0!") == "My phone: 9876543210!"
    
    # Test normal conversation preservation
    assert normalizer.normalize("I have one apple and two oranges.") == "I have 1 apple and 2 oranges."
    assert normalizer.normalize("Order status check please.") == "Order status check please."
