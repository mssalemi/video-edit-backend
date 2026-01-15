import os


class Settings:
    """Application settings loaded from environment variables."""

    DEFAULT_MODEL: str = os.getenv("TRANSCRIBE_DEFAULT_MODEL", "small")
    TMP_DIR: str = os.getenv("TRANSCRIBE_TMP_DIR", "/tmp/transcriber")
    DATA_DIR: str = os.getenv("TRANSCRIBE_DATA_DIR", "/data")


settings = Settings()
