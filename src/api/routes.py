import base64
import time
import logging
import os
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.observability import observe, get_trace_url
from src.asr.whisper_asr import WhisperASR
from src.asr.normalizer import TranscriptNormalizer
from src.nlu.classifier import MockIntentClassifier
from src.nlu.extractor import LLMEntityExtractor
from src.dialogue.manager import DialogueManager, DialogueState
from src.tts.base import MockTTS
from src.core.metrics import metrics_tracker

logger = logging.getLogger("src.api.routes")
router = APIRouter()

# Instantiate pipeline services
asr_service = WhisperASR(model_size=settings.asr.model_name)
transcript_normalizer = TranscriptNormalizer()
nlu_classifier = MockIntentClassifier()
nlu_extractor = LLMEntityExtractor()
dialogue_manager = DialogueManager()
tts_service = MockTTS()

# Start time marker for uptime calculation
START_TIME = time.time()

class TextProcessRequest(BaseModel):
    text: str = Field(..., description="User input text message")
    session_id: str = Field("default", description="Session identifier for state tracking")

class VoiceProcessRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded WAV audio bytes")
    session_id: str = Field("default", description="Session identifier for state tracking")

class VoiceProcessResponse(BaseModel):
    transcript: str
    language: str
    language_probability: float
    intent: str
    intent_confidence: float
    entities: Dict[str, Any]
    dialogue_state: str
    response_text: str
    audio_response_base64: str
    backend_tool_executed: Optional[str] = None
    backend_tool_result: Optional[Dict[str, Any]] = None
    trace_url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    uptime_seconds: float

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Returns application status and diagnostic info."""
    uptime = time.time() - START_TIME
    return HealthResponse(
        status="healthy",
        app_name=settings.app.name,
        version=settings.app.version,
        environment=settings.app.env,
        uptime_seconds=round(uptime, 2)
    )

@router.post("/api/v1/text/process", response_model=VoiceProcessResponse)
@observe(name="text_process_pipeline")
def process_text_message(payload: TextProcessRequest):
    """
    Multimodal Chat Processing:
    Text Input -> Normalizer -> Dialogue Manager -> Backend Tool -> TTS -> Audio Playback.
    """
    logger.info("Text API: Processing chat message for session '%s': '%s'", payload.session_id, payload.text)
    
    raw_transcript = payload.text
    transcript = transcript_normalizer.normalize(raw_transcript)
    if not transcript.strip():
        transcript = "[Silence]"
        
    api_key = os.environ.get("GEMINI_API_KEY")
    dialogue_start = time.time()
    
    intent = "unknown"
    confidence = 1.0
    entities = {}
    
    if api_key:
        logger.info("Text Pipeline: Routing directly to Gemini DialogueManager.")
        dialogue_res = dialogue_manager.process_turn("unknown", {}, transcript, payload.session_id)
        response_text = dialogue_res["response"]
        next_state = dialogue_res["state"]
        tool_executed = dialogue_res["tool_executed"]
        tool_result = dialogue_res["tool_result"]
    else:
        session = dialogue_manager._get_or_create_session(payload.session_id)
        current_state = session.get("state", DialogueState.IDLE)
        
        is_new_request = False
        if current_state != DialogueState.IDLE:
            temp_nlu = nlu_classifier.classify(transcript)
            expected_workflow = session["context"].get("workflow", "unknown")
            if temp_nlu["intent"] != "unknown" and temp_nlu["intent"] != expected_workflow and temp_nlu["confidence"] >= 0.75:
                is_new_request = True
                
        if current_state == DialogueState.IDLE or is_new_request:
            nlu_res = nlu_classifier.classify(transcript)
            intent = nlu_res["intent"]
            confidence = nlu_res["confidence"]
        else:
            intent = session["context"].get("workflow", "unknown")
            confidence = 1.0
            
        entities = nlu_extractor.extract(transcript)
        dialogue_res = dialogue_manager.process_turn(intent, entities, transcript, payload.session_id)
        response_text = dialogue_res["response"]
        next_state = dialogue_res["state"]
        tool_executed = dialogue_res["tool_executed"]
        tool_result = dialogue_res["tool_result"]

    dialogue_latency_ms = (time.time() - dialogue_start) * 1000.0
    
    tool_success = None
    if tool_executed:
        if isinstance(tool_result, dict):
            tool_success = tool_result.get("success", False)
        else:
            tool_success = False
    metrics_tracker.record_turn(asr_ms=0.0, dialogue_ms=dialogue_latency_ms, tool_success=tool_success)
    
    # Synthesize audio for TTS
    try:
        response_audio_bytes = tts_service.synthesize(response_text)
    except Exception as e:
        logger.error("Text Pipeline Error during TTS: %s", e)
        response_audio_bytes = b"RIFFdummybytes"
        
    response_audio_b64 = base64.b64encode(response_audio_bytes).decode("utf-8")
    trace_url = get_trace_url()
    
    return VoiceProcessResponse(
        transcript=transcript,
        language="en",
        language_probability=1.0,
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

@router.post("/api/v1/voice/process", response_model=VoiceProcessResponse)
@observe(name="voice_process_pipeline")
def process_voice_pipeline(payload: VoiceProcessRequest):
    """
    End-to-End Voice AI processing pipeline:
    Base64 Audio Input -> Multilingual ASR -> NLU -> Dialogue Manager -> Backend Tool -> TTS -> Audio Playback.
    """
    logger.info("Voice API: Processing base64 audio payload for session: %s", payload.session_id)
    
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
    
    audio_bytes = await file.read()
    return execute_pipeline(audio_bytes, session_id)

@router.post("/api/v1/telephony/twilio")
def twilio_telephony_webhook(
    SpeechResult: Optional[str] = Form(None),
    CallSid: str = Form("default")
):
    """
    Twilio Telephony Gather Webhook.
    Handles phone caller speech inputs and responds with Gather TwiML XML turns.
    """
    logger.info("Telephony API: Processing call turn. Sid: %s, Input: '%s'", CallSid, SpeechResult)
    
    asr_start = time.time()
    
    if not SpeechResult:
        # Initial greeting when caller dials in
        response_text = "Hello! Welcome to Vani Customer Support. How can I help you today?"
        next_state = "idle"
        tool_executed = None
        tool_result = None
        asr_latency_ms = 0.0
        dialogue_latency_ms = 1.0
    else:
        asr_latency_ms = (time.time() - asr_start) * 1000.0
        
        dialogue_start = time.time()
        # Direct text turn bypasses ASR step
        dialogue_res = dialogue_manager.process_turn("unknown", {}, SpeechResult, CallSid)
        response_text = dialogue_res["response"]
        next_state = dialogue_res["state"]
        tool_executed = dialogue_res["tool_executed"]
        tool_result = dialogue_res["tool_result"]
        dialogue_latency_ms = (time.time() - dialogue_start) * 1000.0
        
        tool_success = None
        if tool_executed:
            if isinstance(tool_result, dict):
                tool_success = tool_result.get("success", False)
            else:
                tool_success = False
        metrics_tracker.record_turn(asr_ms=asr_latency_ms, dialogue_ms=dialogue_latency_ms, tool_success=tool_success)
        
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="en-IN">{response_text}</Say>
    <Gather input="speech" action="/api/v1/telephony/twilio" method="POST" speechTimeout="auto">
        <Say language="en-IN">Please state your next request.</Say>
    </Gather>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@router.get("/api/v1/monitoring/metrics")
def get_system_metrics():
    """Exposes system performance metrics and latencies."""
    return metrics_tracker.get_report()

def execute_pipeline(audio_bytes: bytes, session_id: str) -> VoiceProcessResponse:
    """Helper executing all pipeline steps sequentially with metrics logging."""
    
    # 1. ASR - Automatic Speech Recognition with language detection
    asr_start = time.time()
    try:
        asr_res = asr_service.transcribe_with_meta(audio_bytes)
        raw_transcript = asr_res["text"]
        detected_language = asr_res.get("language", "en")
        language_probability = asr_res.get("language_probability", 1.0)
    except Exception as e:
        logger.error("Voice Pipeline Error during ASR: %s", e)
        raise HTTPException(status_code=422, detail=f"ASR Transcription failed: {e}")
    asr_latency_ms = (time.time() - asr_start) * 1000.0
        
    # Apply Transcript Normalization Layer
    transcript = transcript_normalizer.normalize(raw_transcript)
    if transcript != raw_transcript:
        logger.info("ASR transcript normalized from '%s' to '%s'", raw_transcript, transcript)
        
    # If silence or empty transcript
    if not transcript.strip():
        transcript = "[Silence]"
        
    api_key = os.environ.get("GEMINI_API_KEY")
    dialogue_start = time.time()
    
    intent = "unknown"
    confidence = 1.0
    entities = {}
    
    if api_key:
        logger.info("Optimized Pipeline: Routing directly to Gemini DialogueManager, bypassing local ASR/NLU classifier/extractor steps.")
        dialogue_res = dialogue_manager.process_turn("unknown", {}, transcript, session_id)
        response_text = dialogue_res["response"]
        next_state = dialogue_res["state"]
        tool_executed = dialogue_res["tool_executed"]
        tool_result = dialogue_res["tool_result"]
    else:
        # 2. NLU Intent Classification
        session = dialogue_manager._get_or_create_session(session_id)
        current_state = session.get("state", DialogueState.IDLE)
        
        is_new_request = False
        if current_state != DialogueState.IDLE:
            temp_nlu = nlu_classifier.classify(transcript)
            expected_workflow = session["context"].get("workflow", "unknown")
            if temp_nlu["intent"] != "unknown" and temp_nlu["intent"] != expected_workflow and temp_nlu["confidence"] >= 0.75:
                is_new_request = True
                
        if current_state == DialogueState.IDLE or is_new_request:
            nlu_res = nlu_classifier.classify(transcript)
            intent = nlu_res["intent"]
            confidence = nlu_res["confidence"]
            if is_new_request:
                logger.info("Interruption detected! Switched to new intent: %s", intent)
        else:
            intent = session["context"].get("workflow", "unknown")
            confidence = 1.0
            logger.info(
                "Dialogue session %s is active in state %s. Skipping intent classification and preserving workflow '%s'.",
                session_id, current_state, intent
            )
        
        # 3. NLU Entity Extraction
        entities = nlu_extractor.extract(transcript)
        
        # 4. Dialogue Management & Tool Execution
        dialogue_res = dialogue_manager.process_turn(intent, entities, transcript, session_id)
        response_text = dialogue_res["response"]
        next_state = dialogue_res["state"]
        tool_executed = dialogue_res["tool_executed"]
        tool_result = dialogue_res["tool_result"]

    dialogue_latency_ms = (time.time() - dialogue_start) * 1000.0
    
    # Record metrics
    tool_success = None
    if tool_executed:
        if isinstance(tool_result, dict):
            tool_success = tool_result.get("success", False)
        else:
            tool_success = False
    metrics_tracker.record_turn(asr_ms=asr_latency_ms, dialogue_ms=dialogue_latency_ms, tool_success=tool_success)
    
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
