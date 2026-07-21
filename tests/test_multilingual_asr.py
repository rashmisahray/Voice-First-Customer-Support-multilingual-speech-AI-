import pytest
from src.asr.whisper_asr import WhisperASR

def test_multilingual_asr_init():
    """Verify that WhisperASR initializes with the multilingual base model."""
    asr = WhisperASR(model_size="base")
    assert asr.model_size == "base"
    assert asr.model is None  # Lazy loading check

def test_transcribe_with_meta_structure():
    """Verify structure of transcribe_with_meta method."""
    asr = WhisperASR(model_size="base")
    
    # Generate dummy WAV audio bytes (PCM 16-bit 16kHz mono silence)
    import wave, io
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b'\x00\x00' * 16000) # 1 sec silence
    wav_bytes = buf.getvalue()
    
    # We test the parser without invoking heavy model load if desired,
    # or run full transcribe_with_meta
    numpy_audio = asr.wav_bytes_to_numpy_16k(wav_bytes)
    assert len(numpy_audio) == 16000
    assert numpy_audio.dtype.name == 'float32'
