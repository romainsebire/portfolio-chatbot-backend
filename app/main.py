"""FastAPI application — point d'entrée, CORS, middlewares, routes."""

import hashlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.embeddings import load_model
from app.models import ChatRequest, ChatResponse, ErrorResponse
from app.rag import get_rag_response
from app.security import (
    PromptInjectionError,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    limiter,
    CHAT_RATE_LIMIT,
    sanitize_input,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pré-charger le modèle d'embedding
    logger.info("Starting up — loading embedding model")
    load_model()
    logger.info("Startup complete")
    yield
    # Shutdown
    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Portfolio Chatbot API",
    version="1.0.0",
    docs_url=None,      # Désactivé en production (pas de Swagger public)
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

# --- Rate limiter ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Middlewares (ordre inversé d'exécution: le dernier ajouté est exécuté en premier) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestSizeLimitMiddleware)


# ---------------------------------------------------------------------------
# Exception Handlers (réponses génériques, pas de fuite d'info)
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request. Check your input."},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
@limiter.limit(CHAT_RATE_LIMIT)
async def chat(request: Request, body: ChatRequest):
    # Log le hash du message (traçabilité sans PII)
    msg_hash = hashlib.sha256(body.message.encode()).hexdigest()[:16]
    logger.info("Chat request [hash=%s, lang=%s]", msg_hash, body.lang)

    # Sanitize input
    try:
        clean_message = sanitize_input(body.message)
    except PromptInjectionError:
        logger.warning("Prompt injection blocked [hash=%s]", msg_hash)
        raise HTTPException(status_code=400, detail="Message rejected.")

    # RAG pipeline (agent + MCP)
    thread_id = body.thread_id
    reply = await get_rag_response(clean_message, body.lang, thread_id=thread_id)

    # Résoudre le thread_id effectif (si None, un UUID a été généré)
    effective_thread_id = body.thread_id or msg_hash

    return ChatResponse(reply=reply, thread_id=effective_thread_id)
