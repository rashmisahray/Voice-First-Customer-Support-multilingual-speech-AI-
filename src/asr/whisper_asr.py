import io
import wave
import logging
import numpy as np
from src.core.config import settings

logger = logging.getLogger("src.asr.whisper_asr")

class WhisperASR:
    """ASR implementation using faster-whisper running locally on CPU."""

    def __init__(self, model_size: str = "tiny.en"):
        self.model_size = model_size
        self.model = None
        logger.info("Initializing Whisper ASR with model size: %s", model_size)

    def _load_model(self):
        if self.model is None:
            # Import here to avoid loading overhead during startup
            from faster_whisper import WhisperModel
            logger.info("Loading faster-whisper model '%s' on CPU...", self.model_size)
            # Use INT8 compute type for speed optimization on CPU
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            logger.info("faster-whisper model '%s' loaded successfully.", self.model_size)

    def wav_bytes_to_numpy_16k(self, audio_bytes: bytes) -> np.ndarray:
        """
        Parses standard WAV format bytes directly in memory.
        Bypasses the system FFmpeg requirement by converting the WAV binary
        directly to a 16000Hz float32 mono numpy array.
        """
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                raw_data = wav_file.readframes(n_frames)
                
                # Convert raw byte string to integer numpy array
                if sample_width == 2:
                    data = np.frombuffer(raw_data, dtype=np.int16)
                elif sample_width == 1:
                    data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.int16) - 128
                elif sample_width == 4:
                    data = np.frombuffer(raw_data, dtype=np.int32)
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width} bytes")
                
                # Convert to float32 normalized to [-1.0, 1.0]
                max_val = float(2 ** (8 * sample_width - 1))
                audio_float = data.astype(np.float32) / max_val
                
                # Convert stereo to mono by averaging channels
                if n_channels > 1:
                    logger.info("ASR: Audio has %d channels. Mixing down to mono.", n_channels)
                    audio_float = audio_float.reshape(-1, n_channels).mean(axis=1)
                
                # Resample to 16kHz if different
                if frame_rate != 16000:
                    logger.info("ASR: Resampling audio from %d Hz to 16000 Hz", frame_rate)
                    duration = len(audio_float) / frame_rate
                    new_num_samples = int(duration * 16000)
                    audio_float = np.interp(
                        np.linspace(0, len(audio_float) - 1, new_num_samples),
                        np.arange(len(audio_float)),
                        audio_float
                    ).astype(np.float32)
                
                return audio_float
        except Exception as e:
            logger.error("Failed to parse WAV bytes directly: %s. Attempting fallback.", e)
            raise ValueError(f"Invalid or corrupted WAV file: {e}")

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribes raw WAV audio bytes into text.
        
        Args:
            audio_bytes: In-memory bytes representing a WAV file.
            
        Returns:
            The transcribed text.
        """
        self._load_model()
        
        logger.info("ASR: Transcribing audio file (size: %d bytes)...", len(audio_bytes))
        
        # Convert WAV bytes to 16kHz float32 numpy array
        audio_numpy = self.wav_bytes_to_numpy_16k(audio_bytes)
        
        # Transcribe audio numpy array
        segments, info = self.model.transcribe(audio_numpy, beam_size=5)
        
        # Merge transcript segments
        transcript_text = " ".join([segment.text for segment in segments]).strip()
        
        logger.info("ASR Transcription Complete: '%s' (Detected language: %s)", transcript_text, info.language)
        return transcript_text
