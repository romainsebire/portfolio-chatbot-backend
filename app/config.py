"""Configuration — chargement et validation typée des variables d'environnement."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Pinecone ---
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "portfolio-chatbot"

    # --- Groq ---
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- Embedding ---
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # --- CORS ---
    ALLOWED_ORIGINS: str = "https://romainsebire.fr"

    # --- Sécurité ---
    RATE_LIMIT_PER_MINUTE: int = 10
    MAX_MESSAGE_LENGTH: int = 500

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse la chaîne d'origines séparées par des virgules."""
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]


settings = Settings()  # type: ignore[call-arg]
