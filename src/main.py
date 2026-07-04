import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.core.config import settings
from src.core.logger import setup_logging, logging
from src.api.routes import router

# Setup logging immediately
setup_logging()
logger = logging.getLogger("src.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Starting up %s (version: %s) in %s environment...", 
                settings.app.name, settings.app.version, settings.app.env)
    yield
    # Shutdown actions
    logger.info("Shutting down %s...", settings.app.name)

# Initialize FastAPI application
app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    description="Multilingual Voice AI Customer Support Agent pipeline APIs.",
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, configure exact trusted domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routing
app.include_router(router)

if __name__ == "__main__":
    # In execution, run programmatically if loaded directly
    uvicorn.run(
        "src.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug
    )
