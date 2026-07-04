import abc
import logging

logger = logging.getLogger("src.asr.base")

class BaseASR(abc.ABC):
    """Abstract base class for Speech-to-Text conversion."""

    @abc.abstractmethod
    def transcribe(self, audio_data: bytes) -> str:
        """Transcribes raw audio bytes into text.
        
        Args:
            audio_data: Raw audio data in bytes.
            
        Returns:
            The transcribed text representation.
        """
        pass

class MockASR(BaseASR):
    """Mock ASR that simulates audio transcription."""
    
    def transcribe(self, audio_data: bytes) -> str:
        logger.info("ASR: Received raw audio of size %d bytes", len(audio_data))
        # Simulated transcription of Hinglish/Hindi speech
        mock_transcript = "Yes, main apna order status check karna chahta hoon. Order ID is ORD-876543."
        logger.info("ASR: Transcribed text: %s", mock_transcript)
        return mock_transcript
