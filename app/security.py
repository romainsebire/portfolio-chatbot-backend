"""Security — rate limiting, security headers, input sanitization."""

import re
import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate Limiter (slowapi)
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    storage_uri="memory://",
)

CHAT_RATE_LIMIT = f"{settings.RATE_LIMIT_PER_MINUTE}/minute"

# ---------------------------------------------------------------------------
# Security Headers Middleware
# ---------------------------------------------------------------------------
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": "default-src 'none'",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        return response


# ---------------------------------------------------------------------------
# Request Body Size Limit Middleware
# ---------------------------------------------------------------------------
MAX_BODY_SIZE = 1024  # 1 KB


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_BODY_SIZE:
                return Response(
                    content='{"detail":"Request body too large"}',
                    status_code=413,
                    media_type="application/json",
                )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Input Sanitization
# ---------------------------------------------------------------------------
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_PROMPT_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above|preceding)\s+(instructions|prompts|rules)",
        r"ignore\s+(tes|vos|les)\s+(instructions|consignes|règles)",
        r"(oublie|oublier)\s+(tes|vos|les)\s+(instructions|consignes|règles)",
        r"^system\s*:",
        r"^assistant\s*:",
        r"(you\s+are|tu\s+es)\s+(now|maintenant|désormais)\s+",
        r"(act|behave|respond)\s+(as|like)\s+(a|an|the)\s+",
        r"(agis|comporte|réponds?)\s+(comme|en\s+tant\s+que)\s+",
        r"(reveal|show|display|print)\s+(your|the)\s+(system|initial)\s+(prompt|instructions)",
        r"(montre|affiche|révèle|donne)\s+(ton|le|ta|la)\s+(prompt|instruction|consigne)",
        r"DAN\s+mode",
        r"jailbreak",
        r"do\s+anything\s+now",
    ]
]


class PromptInjectionError(Exception):
    """Raised when a prompt injection attempt is detected in input."""
    pass


def sanitize_input(text: str) -> str:
    """Sanitize user input: strip control chars, detect prompt injection."""
    # Strip control characters
    cleaned = _CONTROL_CHARS_RE.sub("", text).strip()

    # Check prompt injection patterns
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(cleaned):
            logger.warning("Prompt injection attempt detected")
            raise PromptInjectionError("Message rejected")

    return cleaned


def sanitize_output(text: str) -> str:
    """Strip potentially dangerous HTML/script tags from LLM output."""
    # Remove script tags and their content
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove other HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()
