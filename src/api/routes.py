import base64
import time
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from src.core.config import settings
from src.core.logger import logging
from src.asr.base import MockASR
from src.nlu.classifier import MockIntentClassifier
from src.nlu.extractor import MockEntityExtractor
from src.dialogue.manager import DialogueManager, DialogueState
from src.tools.backend_client import BackendClient
from src.tts.base import MockTTS

logger = logging.getLogger("src.api.routes")
router = APIRouter()

# Instantiate singletons/instances for routing
# In real application, we might use dependency injection
asr_service = MockASR()
nlu_classifier = MockIntentClassifier()
nlu_extractor = MockEntityExtractor()
dialogue_manager = DialogueManager()
backend_client = BackendClient()
tts_service = MockTTS()

START_TIME = time.time()

# Request/Response Schemas
class HealthResponse(BaseModel):
    status: str = Field(..., description="Application health status")
    app_name: str = Field(..., description="Name of the application")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Running environment")
    uptime_seconds: float = Field(..., description="Time since application started")

class VoiceProcessRequest(BaseModel):
    audio_base64: str = Field(
        ..., 
        description="Base64 encoded string of user audio input",
        json_schema_extra={"example": "UklGRigAAABXQVZFZm10IBIAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="}
    )

class VoiceProcessResponse(BaseModel):
    transcript: str = Field(..., description="ASR transcribed user text")
    intent: str = Field(..., description="Classified user intent")
    intent_confidence: float = Field(..., description="NLU intent classifier score")
    entities: Dict[str, Any] = Field(..., description="Extracted entities from text")
    dialogue_state: str = Field(..., description="Dialogue manager state after transition")
    response_text: str = Field(..., description="Assistant response text")
    audio_response_base64: str = Field(..., description="Base64 encoded response audio from TTS")
    backend_tool_executed: Optional[str] = Field(None, description="Name of the backend tool executed, if any")
    backend_tool_result: Optional[Dict[str, Any]] = Field(None, description="Result of the backend tool execution")

@router.get("/health", response_model=HealthResponse)
def get_health():
    """Health status checkpoint of Vani services."""
    uptime = time.time() - START_TIME
    logger.info("Health check queried. Uptime: %.2f seconds", uptime)
    return HealthResponse(
        status="healthy",
        app_name=settings.app.name,
        version=settings.app.version,
        environment=settings.app.env,
        uptime_seconds=round(uptime, 2)
    )

@router.post("/api/v1/voice/process-dummy", response_model=VoiceProcessResponse)
def process_dummy_voice(payload: VoiceProcessRequest):
    """
    Dummy pipeline endpoint simulating the voice processing workflow:
    Audio Input -> ASR -> NLU -> Dialogue -> Backend Tool (optional) -> TTS -> Audio Output.
    """
    logger.info("Voice dummy endpoint invoked.")
    
    # 1. Decode raw input audio
    try:
        audio_bytes = base64.b64decode(payload.audio_base64)
    except Exception as e:
        logger.error("Failed to decode base64 audio: %s", e)
        raise HTTPException(status_code=400, detail="Invalid base64 encoded audio data")

    # 2. ASR transcription
    transcript = asr_service.transcribe(audio_bytes)
    
    # 3. NLU processing
    classification = nlu_classifier.classify(transcript)
    intent = classification["intent"]
    confidence = classification["confidence"]
    entities = nlu_extractor.extract(transcript)
    
    # 4. Backend tool call execution (conditional on intent)
    tool_executed = None
    tool_result = None
    
    if intent == "order_status" and "order_id" in entities:
        tool_executed = "get_order_status"
        tool_result = backend_client.get_order_status(entities["order_id"])
        # Update/override dialogue if tool yields response
        if tool_result.get("success"):
            # Provide real data back, preventing hallucinations
            # Let's log it safely (PII masking will redact sensitive logs)
            logger.info("Executed tool get_order_status for %s successfully.", entities["order_id"])
        else:
            logger.warning("Failed to fetch order status for %s.", entities["order_id"])
            
    # 5. Dialogue processing
    dialogue_result = dialogue_manager.process_turn(intent, entities, transcript)
    response_text = dialogue_result["response"]
    next_state = dialogue_result["state"]
    
    # 6. TTS response synthesis
    response_audio_bytes = tts_service.synthesize(response_text)
    response_audio_b64 = base64.b64encode(response_audio_bytes).decode("utf-8")
    
    # 7. Safe logs test (Verify PII masking by writing sensitive content in logs)
    logger.info("Turn Completed. User: '%s'. Intent: '%s'. Entities: %s. Response: '%s'", 
                transcript, intent, entities, response_text)
    
    # Note: sensitive details like email/phones/order IDs in logs will be masked by our PIIMaskingFilter!
    # For example, if the user mentioned a phone number or email, it will be automatically masked.
    
    return VoiceProcessResponse(
        transcript=transcript,
        intent=intent,
        intent_confidence=confidence,
        entities=entities,
        dialogue_state=next_state,
        response_text=response_text,
        audio_response_base64=response_audio_b64,
        backend_tool_executed=tool_executed,
        backend_tool_result=tool_result
    )
