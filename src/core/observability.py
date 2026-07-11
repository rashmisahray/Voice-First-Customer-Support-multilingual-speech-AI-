import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional
from src.core.config import settings

logger = logging.getLogger("src.core.observability")

# Global flag to track if Langfuse tracing is active
LANGFUSE_ACTIVE = False

# Try importing langfuse
try:
    import langfuse
    from langfuse.decorators import observe as lf_observe
    
    # Check if environment variables are set
    has_env = (
        os.environ.get("LANGFUSE_PUBLIC_KEY") is not None and
        os.environ.get("LANGFUSE_SECRET_KEY") is not None
    )
    # Check if config has it (could be loaded from YAML or custom settings)
    has_config = (
        hasattr(settings, "observability") and
        getattr(settings.observability, "public_key", None) and
        getattr(settings.observability, "secret_key", None)
    )
    
    if has_env or has_config:
        LANGFUSE_ACTIVE = True
        logger.info("Langfuse observability successfully initialized for Vani pipeline tracing.")
    else:
        logger.warning(
            "Langfuse keys not detected in environment or config. "
            "Observability tracing will run in Mock/Dry-run mode."
        )
except ImportError:
    logger.warning("Langfuse package not found. Running in mock observability mode.")
    lf_observe = None

def observe(*args, **kwargs) -> Callable:
    """
    Custom wrapper around Langfuse's observe decorator.
    If Langfuse is active, delegates to langfuse.decorators.observe.
    Otherwise, returns a dummy decorator that executes the function normally.
    """
    if LANGFUSE_ACTIVE and lf_observe is not None:
        return lf_observe(*args, **kwargs)
    
    # Mock decorator fallback
    def decorator(func: Callable) -> Callable:
        if asyncio_is_coroutine := getattr(func, "_is_coroutine", None) or hasattr(func, "__code__") and func.__code__.co_flags & 0x80:
            @wraps(func)
            async def async_wrapper(*f_args, **f_kwargs):
                return await func(*f_args, **f_kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*f_args, **f_kwargs):
                return func(*f_args, **f_kwargs)
            return sync_wrapper
            
    # If the decorator is called without arguments (e.g., @observe)
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return decorator(args[0])
        
    return decorator

def get_trace_url() -> Optional[str]:
    """Helper to return active trace UI URL if Langfuse is configured."""
    if not LANGFUSE_ACTIVE:
        return None
    
    host = os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com"
    return f"{host.rstrip('/')}/project"
