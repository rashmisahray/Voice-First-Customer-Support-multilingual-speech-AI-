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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os

# Register routing
app.include_router(router)

# Mount static files for HTML UI frontend
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
def redirect_to_frontend():
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    # In execution, run programmatically if loaded directly
    uvicorn.run(
        "src.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug
    )
