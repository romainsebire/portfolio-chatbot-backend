"""FastAPI application: entry point, CORS, middlewares, and routes."""

from contextlib import asynccontextmanager
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import os
import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.models import ChatRequest, ChatResponse, ErrorResponse
from app.rag import get_rag_response
from app.security import (
    PromptInjectionError,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    limiter,
    CHAT_RATE_LIMIT,
    sanitize_input,
    token_limiter,
    current_client_ip,
)

# Logging configuration (Console stdout + Rotating file)
log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

handlers: list[logging.Handler] = [logging.StreamHandler()]

try:
    log_file = os.environ.get("LOG_FILE_PATH", "app.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=20 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(file_handler)
except Exception:
    pass

logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
logger = logging.getLogger(__name__)


# Lifespan (startup / shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: initializing MCP")
    from app.rag import init_mcp
    await init_mcp()
    logger.info("Startup complete")
    yield
    logger.info("Shutting down")
    from app.rag import close_mcp
    await close_mcp()


# FastAPI App
app = FastAPI(
    title="Portfolio Chatbot API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

app.state.limiter = limiter


async def _get_request_lang(request: Request) -> str:
    """Extract language preference ('fr' or 'en') from request JSON body or headers."""
    try:
        body = await request.json()
        if isinstance(body, dict) and body.get("lang") == "en":
            return "en"
    except Exception:
        pass
    accept_lang = request.headers.get("accept-language", "")
    if accept_lang.startswith("en"):
        return "en"
    return "fr"


@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    lang = await _get_request_lang(request)
    if lang == "en":
        detail = "You are sending messages too quickly. Please wait a minute before trying again."
    else:
        detail = "Vous envoyez des messages trop rapidement. Veuillez patienter une minute avant de réessayer."
    return JSONResponse(
        status_code=429,
        content={"detail": detail},
    )


# Middlewares
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    lang = await _get_request_lang(request)
    if lang == "en":
        detail = "The message format is invalid or exceeds the allowed length limit (250 characters max)."
    else:
        detail = "Le format du message est invalide ou dépasse la limite autorisée (250 caractères max)."
    return JSONResponse(
        status_code=422,
        content={"detail": detail},
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
    lang = await _get_request_lang(request)
    if lang == "en":
        detail = "A temporary technical error occurred while processing your request. Please try again in a few moments."
    else:
        detail = "Une erreur technique temporaire est survenue lors du traitement de votre demande. Veuillez réessayer dans quelques instants."
    return JSONResponse(
        status_code=500,
        content={"detail": detail},
    )


# Routes
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
    client_ip = get_remote_address(request)
    current_client_ip.set(client_ip)

    if token_limiter.is_ip_blocked(client_ip):
        logger.warning("Blocked IP access attempt: %s", client_ip)
        raise HTTPException(status_code=403, detail="Access denied.")

    token_limiter.add_request(client_ip)

    if not token_limiter.check_limit(client_ip):
        used = token_limiter.get_used_tokens(client_ip)
        logger.warning("Daily token limit reached for IP %s (%d/%d)", client_ip, used, settings.MAX_DAILY_TOKENS_PER_IP)
        if body.lang == "fr":
            detail_msg = (
                f"Vous avez atteint la limite quotidienne de tokens pour votre adresse IP ({used} / {settings.MAX_DAILY_TOKENS_PER_IP} tokens).\n\n"
                "Ce chatbot est une démonstration technique personnelle dont l'utilisation est volontairement limitée afin de préserver l'infrastructure. "
                "Merci de votre visite et à demain !"
            )
        else:
            detail_msg = (
                f"You have reached the daily token quota for your IP address ({used} / {settings.MAX_DAILY_TOKENS_PER_IP} tokens).\n\n"
                "This chatbot is a personal technical demonstration with limited daily quotas for infrastructure reasons. "
                "Thank you for visiting, and see you tomorrow!"
            )
        raise HTTPException(
            status_code=429,
            detail=detail_msg,
        )

    msg_hash = hashlib.sha256(body.message.encode()).hexdigest()[:16]
    logger.info("Chat request [IP=%s, hash=%s, lang=%s]", client_ip, msg_hash, body.lang)

    try:
        clean_message = sanitize_input(body.message)
    except PromptInjectionError:
        logger.warning("Prompt injection blocked [hash=%s]", msg_hash)
        detail_msg = (
            "Votre message n'a pas pu être traité car il contient des consignes ou expressions non autorisées. Merci de reformuler votre question."
            if body.lang == "fr"
            else "Your message could not be processed because it contains unauthorized instructions. Please rephrase your question."
        )
        raise HTTPException(status_code=400, detail=detail_msg)

    thread_id = body.thread_id
    reply, effective_thread_id = await get_rag_response(clean_message, body.lang, thread_id=thread_id)

    return ChatResponse(reply=reply, thread_id=effective_thread_id)


@app.get("/admin/daily-stats")
@limiter.limit("5/minute")
async def get_daily_stats(request: Request, key: str = ""):
    """Secured admin endpoint for inspecting daily usage statistics."""
    header_key = request.headers.get("x-admin-key", "")
    provided_key = header_key or key
    if not provided_key or not secrets.compare_digest(provided_key, settings.ADMIN_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing admin key.")
    return token_limiter.get_stats_summary()
