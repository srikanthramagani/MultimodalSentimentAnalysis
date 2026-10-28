# MIT License
# Multimodal Sentiment Analysis - FastAPI Application

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.models import image_analyzer, text_analyzer, voice_analyzer
from app.routers import sentiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading text model...")
    text_analyzer.load_model()
    logger.info("Loading image model...")
    image_analyzer.load_model()
    logger.info("Loading voice model...")
    voice_analyzer.load_model()
    logger.info("All models loaded. Ready to serve.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Multimodal Sentiment Analysis",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8002", "http://127.0.0.1:8002"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

app.include_router(sentiment.router)

# Serve frontend static files at root
_frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
