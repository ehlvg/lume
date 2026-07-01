from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", env_prefix="LUME_", extra="ignore"
    )

    ollama_base_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4:12b-mlx"
    bind_host: str = "0.0.0.0"
    bind_port: int = 8000

    # Admin bootstrap key — present in env only on first launch to seed the keys table.
    admin_bootstrap_key: str | None = None

    # Voice binaries
    whisper_bin: str = os.path.expanduser("~/.local/bin/whisper")
    whisper_model: str = "turbo"
    whisper_device: str = "cpu"
    supertonic_enabled: bool = True
    supertonic_voice: str = "M1"
    supertonic_steps: int = 8
    supertonic_speed: float = 1.0
    supertonic_lang: str = "na"  # language-agnostic auto-detect

    # Tool loop
    max_tool_iterations: int = 6

    # Paths
    data_dir: Path = ROOT / "data"
    attachments_dir: Path = Field(default_factory=lambda: ROOT / "data" / "attachments")
    db_path: Path = Field(default_factory=lambda: ROOT / "data" / "lume.db")
    voice_dir: Path = Field(default_factory=lambda: ROOT / "data" / "voice")

    @property
    def db_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.db_path}"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.attachments_dir.mkdir(parents=True, exist_ok=True)
    s.voice_dir.mkdir(parents=True, exist_ok=True)
    return s