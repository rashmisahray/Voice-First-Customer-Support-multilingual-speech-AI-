import os
from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseModel):
    name: str = "Vani Voice AI Support"
    version: str = "1.0.0"
    env: str = "development"

class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True

class LoggingConfig(BaseModel):
    level: str = "INFO"
    mask_pii: bool = True
    pii_mask_char: str = "*"
    log_to_file: bool = True
    log_file_path: str = "logs/vani.log"

class ASRConfig(BaseModel):
    model_name: str = "whisper-turbo"
    language: str = "multilingual"

class NLUConfig(BaseModel):
    intent_confidence_threshold: float = 0.6
    max_intents: int = 50

class DialogueConfig(BaseModel):
    session_timeout_seconds: int = 300
    enable_llm_fallback: bool = True

class TTSConfig(BaseModel):
    model_name: str = "xtts"
    voice: str = "female_calm"

class Settings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    nlu: NLUConfig = Field(default_factory=NLUConfig)
    dialogue: DialogueConfig = Field(default_factory=DialogueConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore"
    )

def load_settings(config_path: str | Path = None) -> Settings:
    """Loads configuration from YAML, then overrides with environment variables."""
    if not config_path:
        # Default path relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "configs" / "config.yaml"

    yaml_data: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    yaml_data = loaded
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}. Using defaults.")

    # Initialize settings from dictionary
    return Settings(**yaml_data)

# Singleton settings instance for the app
settings = load_settings()
