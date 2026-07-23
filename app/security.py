"""Security: rate limiting, security headers, input/output sanitization."""

import contextvars
import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

logger = logging.getLogger(__name__)

# Context variable to propagate current client IP to LLM callbacks
current_client_ip: contextvars.ContextVar[str] = contextvars.ContextVar("current_client_ip", default="127.0.0.1")

stats_logger = logging.getLogger("daily_stats")


class DailyTokenLimiter:
    """Daily token quota manager per IP and daily usage statistics with 90-day retention."""

    def __init__(
        self,
        max_daily_tokens: int,
        stats_file: str = "daily_stats.json",
        stats_log_file: str = "daily_stats.log",
        retention_days: int = 90,
    ):
        self.max_daily_tokens = max_daily_tokens
        self.stats_file = stats_file
        self.stats_log_file = stats_log_file
        self.retention_days = retention_days
        self.history: dict[str, dict] = self._load_history()
        self._purge_old_entries()
        self._setup_stats_logger()

    def _setup_stats_logger(self):
        """Configure lightweight dedicated file handler for daily summary logging."""
        try:
            handler = logging.FileHandler(self.stats_log_file, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
            stats_logger.addHandler(handler)
            stats_logger.setLevel(logging.INFO)
            stats_logger.propagate = False
        except Exception:
            pass

    def _load_history(self) -> dict[str, dict]:
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {self.stats_file}: {e}")
        return {}

    def _save_history(self):
        try:
            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Error writing to {self.stats_file}: {e}")

    def _purge_old_entries(self):
        """Purge entries and log lines older than retention threshold (default 90 days)."""
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            cutoff_str = cutoff.strftime("%Y-%m-%d")

            initial_count = len(self.history)
            self.history = {
                date_str: data for date_str, data in self.history.items()
                if date_str >= cutoff_str
            }
            if len(self.history) != initial_count:
                self._save_history()

            if os.path.exists(self.stats_log_file):
                filtered_lines = []
                with open(self.stats_log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", line)
                        if match:
                            if match.group(1) >= cutoff_str:
                                filtered_lines.append(line)
                        else:
                            filtered_lines.append(line)

                with open(self.stats_log_file, "w", encoding="utf-8") as f:
                    f.writelines(filtered_lines)
        except Exception as e:
            logger.warning(f"Error purging log entries older than {self.retention_days} days: {e}")

    def _get_today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def is_ip_blocked(self, ip: str) -> bool:
        """Check whether IP is present in configured blacklist."""
        blocked_list = [i.strip() for i in settings.BLOCKED_IPS.split(",") if i.strip()]
        return ip in blocked_list

    def check_limit(self, ip: str) -> bool:
        today = self._get_today_str()
        day_stats = self.history.get(today, {}).get("users", {})
        user_info = day_stats.get(ip, {})
        return user_info.get("tokens", 0) < self.max_daily_tokens

    def get_used_tokens(self, ip: str) -> int:
        today = self._get_today_str()
        day_stats = self.history.get(today, {}).get("users", {})
        user_info = day_stats.get(ip, {})
        return user_info.get("tokens", 0)

    def add_request(self, ip: str):
        today = self._get_today_str()
        if today not in self.history:
            self._purge_old_entries()
            self.history[today] = {
                "date": today,
                "total_requests": 0,
                "total_tokens": 0,
                "total_users": 0,
                "users": {},
            }

        day_entry = self.history[today]
        day_entry["total_requests"] += 1

        users = day_entry["users"]
        if ip not in users:
            users[ip] = {"requests": 1, "tokens": 0}
            day_entry["total_users"] = len(users)
        else:
            users[ip]["requests"] += 1

        self._save_history()

    def add_tokens(self, ip: str, tokens: int):
        today = self._get_today_str()
        if today not in self.history:
            self.add_request(ip)

        day_entry = self.history[today]
        day_entry["total_tokens"] += tokens

        users = day_entry["users"]
        if ip not in users:
            users[ip] = {"requests": 1, "tokens": tokens}
            day_entry["total_users"] = len(users)
        else:
            users[ip]["tokens"] += tokens

        self._save_history()

        stats_logger.info(
            f"SUMMARY {today} | IP: {ip} | +{tokens} tokens | IP Total: {users[ip]['tokens']} tokens ({users[ip]['requests']} req) | Day Total: {day_entry['total_tokens']} tokens ({day_entry['total_requests']} req, {day_entry['total_users']} IPs)"
        )
        logger.info(f"[TOKEN RATELIMIT] IP {ip} -> +{tokens} tokens (Today total: {users[ip]['tokens']}/{self.max_daily_tokens})")

    def get_stats_summary(self) -> dict:
        return self.history


token_limiter = DailyTokenLimiter(max_daily_tokens=settings.MAX_DAILY_TOKENS_PER_IP)

# Rate Limiter (slowapi)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    storage_uri="memory://",
)

CHAT_RATE_LIMIT = f"{settings.RATE_LIMIT_PER_MINUTE}/minute"

# Security Headers Middleware
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


# Request Body Size Limit Middleware
MAX_BODY_SIZE = 4096


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_BODY_SIZE:
                return Response(
                    content='{"detail":"La taille de la requête dépasse la limite autorisée (1 Ko max). / Request payload exceeds maximum allowed size (1 KB max)."}',
                    status_code=413,
                    media_type="application/json",
                )
        return await call_next(request)


# Input Sanitization
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
    """Sanitize user input: strip control chars and detect prompt injection patterns."""
    cleaned = _CONTROL_CHARS_RE.sub("", text).strip()

    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(cleaned):
            logger.warning("Prompt injection attempt detected")
            raise PromptInjectionError("Message rejected")

    return cleaned


def sanitize_output(text: str) -> str:
    """Strip potentially dangerous HTML/script tags from LLM output."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()
