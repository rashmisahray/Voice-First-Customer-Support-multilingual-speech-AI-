import base64
import time
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.observability import observe, get_trace_url
from src.asr.whisper_asr import WhisperASR
from src.asr.normalizer import TranscriptNormalizer
from src.nlu.classifier import MockIntentClassifier
from src.nlu.extractor import LLMEntityExtractor
from src.dialogue.manager import DialogueManager
from src.tts.base import MockTTS

logger = logging.getLogger("src.api.routes")
router = APIRouter()

# Instantiate pipeline services
asr_service = WhisperASR(model_size=settings.asr.model_name)
transcript_normalizer = TranscriptNormalizer()
nlu_classifier = MockIntentClassifier()
nlu_extractor = LLMEntityExtractor()
dialogue_manager = DialogueManager()
tts_service = MockTTS()

START_TIME = time.time()

# Request/Response Schemas
class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    uptime_seconds: float

class VoiceProcessRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded WAV audio data")
    session_id: str = Field("default", description="Session identifier for multi-turn conversations")

class VoiceProcessResponse(BaseModel):
    transcript: str
    language: Optional[str] = "en"
    language_probability: Optional[float] = 1.0
    intent: str
    intent_confidence: float
    entities: Dict[str, Any]
    dialogue_state: str
    response_text: str
    audio_response_base64: str
    backend_tool_executed: Optional[str] = None
    backend_tool_result: Optional[Dict[str, Any]] = None
    trace_url: Optional[str] = None

@router.get("/health", response_model=HealthResponse)
def get_health():
    """Health status checkpoint."""
    uptime = time.time() - START_TIME
    return HealthResponse(
        status="healthy",
        app_name=settings.app.name,
        version=settings.app.version,
        environment=settings.app.env,
        uptime_seconds=round(uptime, 2)
    )

@router.post("/api/v1/voice/process", response_model=VoiceProcessResponse)
@observe(name="voice_process_pipeline")
def process_voice_pipeline(payload: VoiceProcessRequest):
    """
    End-to-End Voice AI processing pipeline:
    Base64 Audio Input -> Multilingual ASR -> NLU -> Dialogue Manager -> Backend Tool -> TTS -> Audio Playback.
    """
    logger.info("Voice API: Processing base64 audio payload for session: %s", payload.session_id)
    
    # 1. Decode base64 audio string to raw WAV bytes
    try:
        audio_bytes = base64.b64decode(payload.audio_base64)
    except Exception as e:
        logger.error("Voice API: Failed to decode base64 payload: %s", e)
        raise HTTPException(status_code=400, detail="Invalid base64 encoded audio data")
        
    return execute_pipeline(audio_bytes, payload.session_id)

@router.post("/api/v1/voice/upload", response_model=VoiceProcessResponse)
@observe(name="voice_upload_pipeline")
async def process_voice_upload(
    file: UploadFile = File(..., description="Uploaded WAV audio file"),
    session_id: str = Form("default", description="Session identifier")
):
    """
    End-to-End Voice AI pipeline accepting a direct audio file upload.
    """
    logger.info("Voice API: Processing uploaded file '%s' for session: %s", file.filename, session_id)
    
    # Read audio bytes
    audio_bytes = await file.read()
    return execute_pipeline(audio_bytes, session_id)

def execute_pipeline(audio_bytes: bytes, session_id: str) -> VoiceProcessResponse:
    """Helper executing all pipeline steps sequentially."""
    
    # 1. ASR - Automatic Speech Recognition with language detection
    try:
        asr_res = asr_service.transcribe_with_meta(audio_bytes)
        raw_transcript = asr_res["text"]
        detected_language = asr_res.get("language", "en")
        language_probability = asr_res.get("language_probability", 1.0)
    except Exception as e:
        logger.error("Voice Pipeline Error during ASR: %s", e)
        raise HTTPException(status_code=422, detail=f"ASR Transcription failed: {e}")
        
    # Apply Transcript Normalization Layer
    transcript = transcript_normalizer.normalize(raw_transcript)
    if transcript != raw_transcript:
        logger.info("ASR transcript normalized from '%s' to '%s'", raw_transcript, transcript)
        
    # If silence or empty transcript
    if not transcript.strip():
        transcript = "[Silence]"
        
    # 2. NLU Intent Classification
    nlu_res = nlu_classifier.classify(transcript)
    intent = nlu_res["intent"]
    confidence = nlu_res["confidence"]
    
    # 3. NLU Entity Extraction
    entities = nlu_extractor.extract(transcript)
    
    # 4. Dialogue Management & Tool Execution
    dialogue_res = dialogue_manager.process_turn(intent, entities, transcript, session_id)
    response_text = dialogue_res["response"]
    next_state = dialogue_res["state"]
    tool_executed = dialogue_res["tool_executed"]
    tool_result = dialogue_res["tool_result"]
    
    # 5. TTS Response Synthesis (convert text to audio)
    try:
        response_audio_bytes = tts_service.synthesize(response_text)
    except Exception as e:
        logger.error("Voice Pipeline Error during TTS: %s", e)
        response_audio_bytes = b"RIFFdummybytes"
        
    response_audio_b64 = base64.b64encode(response_audio_bytes).decode("utf-8")
    
    # Get Langfuse trace URL if enabled
    trace_url = get_trace_url()
    
    logger.info("Voice Pipeline: Completed turn successfully for session %s (Language: %s).", session_id, detected_language)
    return VoiceProcessResponse(
        transcript=transcript,
        language=detected_language,
        language_probability=language_probability,
        intent=intent,
        intent_confidence=confidence,
        entities=entities,
        dialogue_state=next_state,
        response_text=response_text,
        audio_response_base64=response_audio_b64,
        backend_tool_executed=tool_executed,
        backend_tool_result=tool_result,
        trace_url=trace_url
    )
