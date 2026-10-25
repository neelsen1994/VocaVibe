"""Runtime configuration.

Every setting is sourced from environment variables so one container image can
be promoted unchanged across environments (dev -> staging -> prod). On AWS the
values are injected by the ECS task definition / Lambda configuration, with
secrets such as ``OPENAI_API_KEY`` resolved from Secrets Manager or SSM
Parameter Store rather than baked into the image.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Local development only: real environment variables always win over the file.
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

LLMProvider = Literal["auto", "anthropic", "openai", "offline"]
_PROVIDERS = ("auto", "anthropic", "openai", "offline")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"Environment variable {name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"Environment variable {name} must be within [{minimum}, {maximum}], got {value}")
    return value


# No ``slots=True``: get_settings() reads the defaults back as class attributes.
@dataclass(frozen=True)
class Settings:
    """Immutable application settings. Defaults are safe for local development."""

    app_name: str = "VocaVibe German Learning API"
    # "auto" picks Anthropic if its key is set, then OpenAI, else deterministic
    # offline mode (curated lexicon + hashing embeddings, no network access).
    llm_provider: LLMProvider = "auto"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    # Claude Opus 5.5 defaults to "medium"; set explicitly so behaviour is pinned.
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    # OpenAI only: Claude's current models reject sampling parameters.
    llm_temperature: float = 0.2
    # Thinking models can take a while on a full dictionary entry.
    llm_timeout_seconds: float = 120.0
    # Number of times the grammar critic may send a draft back for repair.
    max_critique_retries: int = 2
    flashcard_ttl_days: int = 180
    history_ttl_days: int = 365
    # Minimum similarity (0..1) for a spoken/typed answer to count as correct.
    fuzzy_match_threshold: float = 0.85
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)

    @property
    def active_provider(self) -> Literal["anthropic", "openai", "offline"]:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "offline"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings from the environment (and ``.env`` if present) once per process."""
    load_dotenv(ENV_FILE, override=False)

    provider = _env_str("LLM_PROVIDER", Settings.llm_provider).lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {_PROVIDERS}, got {provider!r}")
    effort = _env_str("ANTHROPIC_EFFORT", Settings.anthropic_effort).lower()
    if effort not in ("low", "medium", "high", "xhigh", "max"):
        raise ValueError(f"ANTHROPIC_EFFORT must be low|medium|high|xhigh|max, got {effort!r}")

    return Settings(
        app_name=_env_str("APP_NAME", Settings.app_name),
        llm_provider=provider,  # type: ignore[arg-type]
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip() or None,
        anthropic_model=_env_str("ANTHROPIC_MODEL", Settings.anthropic_model),
        anthropic_effort=effort,  # type: ignore[arg-type]
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip() or None,
        openai_model=_env_str("OPENAI_MODEL", Settings.openai_model),
        openai_embedding_model=_env_str("OPENAI_EMBEDDING_MODEL", Settings.openai_embedding_model),
        llm_temperature=_env_float("LLM_TEMPERATURE", Settings.llm_temperature, minimum=0.0, maximum=2.0),
        llm_timeout_seconds=_env_float(
            "LLM_TIMEOUT_SECONDS", Settings.llm_timeout_seconds, minimum=1.0, maximum=300.0
        ),
        max_critique_retries=_env_int("MAX_CRITIQUE_RETRIES", Settings.max_critique_retries),
        flashcard_ttl_days=_env_int("FLASHCARD_TTL_DAYS", Settings.flashcard_ttl_days, minimum=1),
        history_ttl_days=_env_int("HISTORY_TTL_DAYS", Settings.history_ttl_days, minimum=1),
        fuzzy_match_threshold=_env_float(
            "FUZZY_MATCH_THRESHOLD", Settings.fuzzy_match_threshold, minimum=0.0, maximum=1.0
        ),
        cors_origins=tuple(  # comma-separated list
            origin.strip()
            for origin in _env_str("CORS_ORIGINS", ",".join(Settings.cors_origins)).split(",")
            if origin.strip()
        ),
    )
