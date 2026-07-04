import abc
import logging

logger = logging.getLogger("src.tts.base")

class BaseTTS(abc.ABC):
    """Abstract base class for Text-to-Speech generation."""

    @abc.abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Synthesizes input text into raw audio bytes.
        
        Args:
            text: Response message string to synthesize.
            
        Returns:
            Raw audio data in bytes (e.g. WAV format).
        """
        pass

class MockTTS(BaseTTS):
    """Mock TTS that generates dummy audio bytes for testing purposes."""
    
    def synthesize(self, text: str) -> bytes:
        logger.info("TTS: Synthesizing voice response for text: '%s'", text)
        # Generate dummy byte array representing an audio output stream
        dummy_audio = b"RIFFmockaudiobytesdataforvanioutput"
        logger.info("TTS: Synthesized audio stream size: %d bytes", len(dummy_audio))
        return dummy_audio
