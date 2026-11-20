"""Application configuration: loading and typed validation of environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "portfolio-chatbot"

    # Groq
    GROQ_API_KEY: str
    GROQ_CASCADE_MODELS: str = (
        "openai/gpt-oss-120b,"
        "qwen/qwen3.8-27b,"
        "openai/gpt-oss-20b"
    )

    @property
    def groq_cascade_list(self) -> list[str]:
        """Return the ordered list of models for the Groq fallback cascade."""
        return [m.strip() for m in self.GROQ_CASCADE_MODELS.split(",") if m.strip()]

    # Embedding
    EMBEDDING_MODEL: str = "multilingual-e5-large"

    # CORS
    ALLOWED_ORIGINS: str = "https://romainsebire.fr"

    # Security & Rate Limits
    RATE_LIMIT_PER_MINUTE: int = 10
    MAX_DAILY_TOKENS_PER_IP: int = 50000
    MAX_MESSAGE_LENGTH: int = 250
    ADMIN_KEY: str = "admin-secret-key-change-me"
    BLOCKED_IPS: str = ""

    # RAG Parameters
    MIN_SIMILARITY_SCORE: float = 0.79

    # Logging
    LOG_LEVEL: str = "INFO"

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse comma-separated origin strings."""
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]


settings = Settings()  # type: ignore[call-arg]
