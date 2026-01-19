import os


class Settings:
    """Application settings loaded from environment variables."""

    DEFAULT_MODEL: str = os.getenv("TRANSCRIBE_DEFAULT_MODEL", "small")
    TMP_DIR: str = os.getenv("TRANSCRIBE_TMP_DIR", "/tmp/transcriber")
    DATA_DIR: str = os.getenv("TRANSCRIBE_DATA_DIR", "/data")

    # AI Planner settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Chunked transcription settings
    MAX_SEGMENT_MS: int = int(os.getenv("MAX_SEGMENT_MS", "3000"))


settings = Settings()
