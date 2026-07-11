import abc
import os
import logging
import io

logger = logging.getLogger("src.tts.base")

class BaseTTS(abc.ABC):
    """Abstract base class for Text-to-Speech generation."""

    @abc.abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Synthesizes input text into raw audio bytes.
        
        Args:
            text: Response message string to synthesize.
            
        Returns:
            Raw audio data in bytes.
        """
        pass

class MockTTS(BaseTTS):
    """
    Production-ready TTS service with support for:
    1. ElevenLabs API (if ELEVENLABS_API_KEY is configured in env)
    2. gTTS (Google Text-to-Speech) as a zero-key, high-quality fallback
    3. Mock dummy bytes (as local fallback if network is offline)
    """
    
    def __init__(self):
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        if self.api_key:
            logger.info("ElevenLabs API Key detected. Using ElevenLabs as primary TTS engine.")
        else:
            logger.info("No ElevenLabs API key found. Using gTTS (Google TTS) as primary TTS engine.")

    def synthesize(self, text: str) -> bytes:
        logger.info("TTS: Synthesizing voice response for text: '%s'", text)
        
        # 1. ElevenLabs Synthesis (if API key is present)
        if self.api_key:
            try:
                from elevenlabs.client import ElevenLabs
                client = ElevenLabs(api_key=self.api_key)
                
                # Perform the generation
                audio_generator = client.generate(
                    text=text,
                    voice="Rachel",
                    model="eleven_monolingual_v1"
                )
                # Convert generator to bytes
                audio_bytes = b"".join(audio_generator)
                logger.info("TTS: Successfully synthesized voice using ElevenLabs (%d bytes)", len(audio_bytes))
                return audio_bytes
            except Exception as e:
                logger.error("ElevenLabs TTS failed: %s. Falling back to gTTS.", e)

        # 2. gTTS Synthesis (Zero-key fallback)
        try:
            from gtts import gTTS
            fp = io.BytesIO()
            # Generate English speech
            tts = gTTS(text=text, lang="en", slow=False)
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_bytes = fp.read()
            logger.info("TTS: Successfully synthesized voice using gTTS fallback (%d bytes)", len(audio_bytes))
            return audio_bytes
        except Exception as e:
            logger.error("gTTS Synthesis failed: %s. Falling back to offline dummy audio.", e)

        # 3. Offline Mock audio fallback (so it never crashes)
        dummy_audio = b"RIFFmockaudiobytesdataforvanioutput"
        logger.info("TTS: Offline fallback used (%d bytes)", len(dummy_audio))
        return dummy_audio
