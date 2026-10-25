"""FastAPI entrypoint and HTTP routes.

Run locally:  uvicorn app.main:app --reload

The process is stateless apart from the in-memory mocks, so in production it
scales horizontally behind an ALB / API Gateway (ECS Fargate or Lambda with
an ASGI adapter) once the mocks are swapped for DynamoDB and OpenSearch
Serverless.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.graph.state import CompiledStateGraph

from app import __version__
from app.agents.graph import build_dictionary_graph
from app.agents.state import initial_state
from app.config import Settings, get_settings
from app.schemas.api import (
    AnalyzeRequest,
    AnalyzeResponse,
    FlashcardAttemptRequest,
    FlashcardAttemptResponse,
    FlashcardSaveRequest,
    FlashcardSaveResponse,
    PracticeResponse,
    UserId,
    WordCandidateOut,
)
from app.schemas.dictionary import DictionaryResponse
from app.services.llm_service import LLMService, LLMServiceError, LLMUnavailableError
from app.services.practice_service import PracticeService
from app.services.storage_service import (
    DuplicateItemError,
    FlashcardRepository,
    ItemNotFoundError,
    MockDynamoDB,
    MockVectorStore,
    StorageError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServiceContainer:
    settings: Settings
    llm: LLMService
    db: MockDynamoDB
    vectors: MockVectorStore
    flashcards: FlashcardRepository
    practice: PracticeService
    graph: CompiledStateGraph


def build_services(settings: Settings) -> ServiceContainer:
    llm = LLMService(settings)
    db = MockDynamoDB()
    vectors = MockVectorStore(llm.build_embedder())
    flashcards = FlashcardRepository(
        db, flashcard_ttl_days=settings.flashcard_ttl_days, history_ttl_days=settings.history_ttl_days
    )
    return ServiceContainer(
        settings=settings,
        llm=llm,
        db=db,
        vectors=vectors,
        flashcards=flashcards,
        practice=PracticeService(flashcards, vectors, llm, fuzzy_threshold=settings.fuzzy_match_threshold),
        graph=build_dictionary_graph(llm, max_retries=settings.max_critique_retries),
    )


def get_services(request: Request) -> ServiceContainer:
    return request.app.state.services


Services = Annotated[ServiceContainer, Depends(get_services)]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

analysis_router = APIRouter(prefix="/api", tags=["analysis"])
flashcard_router = APIRouter(prefix="/api/flashcard", tags=["flashcards"])


@analysis_router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest, services: Services) -> AnalyzeResponse:
    """Run the agent graph on a typed word or a voice-transcribed phrase.

    Voice input that is not an exact known word returns ``needs_confirmation``
    with ranked candidates; re-submit the learner's pick with ``confirmed=true``.
    """
    state = initial_state(
        body.text, source=body.source, confirmed=body.confirmed, requested_tense=body.requested_tense
    )
    # Graph nodes make blocking SDK calls; keep them off the event loop.
    final = await run_in_threadpool(services.graph.invoke, state)

    if final["analyzed_intent"] == "invalid":
        raise HTTPException(
            status_code=422,
            detail=final.get("rejection_reason") or "Input is not a recognisable German word or phrase.",
        )

    candidates = [WordCandidateOut.model_validate(c) for c in final.get("candidates", [])]
    common = {
        "input": body.text,
        "intent": final["analyzed_intent"],
        "root_form": final.get("root_form", body.text),
        "is_idiom": final.get("is_idiom", False),
        "iterations": final["iteration_count"],
        "candidates": candidates,
        "llm_mode": services.llm.mode,
    }
    if final.get("needs_confirmation"):
        return AnalyzeResponse(status="needs_confirmation", critique_errors=[], result=None, **common)

    errors = final["critique_errors"]
    payload = final["validated_json"] or final["raw_dictionary_data"]
    return AnalyzeResponse(
        status="unverified" if errors else "validated",
        critique_errors=errors,
        result=DictionaryResponse.model_validate(payload),
        **common,
    )


@flashcard_router.post("/save", response_model=FlashcardSaveResponse, status_code=status.HTTP_201_CREATED)
async def save_flashcard(body: FlashcardSaveRequest, services: Services) -> FlashcardSaveResponse:
    """Persist ONE sense of a word as its own flashcard (409 if that sense is already saved)."""
    card = await run_in_threadpool(
        services.flashcards.save_flashcard,
        user_id=body.user_id,
        word=body.word,
        part_of_speech=body.part_of_speech,
        context=body.context.model_dump(mode="json"),
        conjugations=body.conjugations.model_dump(mode="json") if body.conjugations else None,
        ttl_days=body.ttl_days,
    )
    return FlashcardSaveResponse(
        card_id=card["card_id"],
        user_id=card["user_id"],
        word=card["word"],
        sense=card["sense"],
        created_at=card["created_at"],
        expires_at=card["expires_at"],
    )


@flashcard_router.post("/attempt", response_model=FlashcardAttemptResponse)
async def record_attempt(body: FlashcardAttemptRequest, services: Services) -> FlashcardAttemptResponse:
    """Score a typed or spoken answer to a practice exercise; mistakes feed the practice RAG index."""
    return await run_in_threadpool(
        services.practice.record_attempt,
        user_id=body.user_id,
        exercise_id=body.exercise_id,
        transcript=body.transcript,
    )


@flashcard_router.get("/practice", response_model=PracticeResponse)
async def practice(
    services: Services,
    user_id: Annotated[UserId, Query()],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> PracticeResponse:
    """Personalised exercises retrieved from the learner's past mistakes."""
    return await run_in_threadpool(services.practice.build_practice_set, user_id=user_id, limit=limit)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": message})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = build_services(settings)
        logger.info("Services ready (llm_mode=%s)", app.state.services.llm.mode)
        yield

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(analysis_router)
    app.include_router(flashcard_router)

    @app.get("/health", tags=["ops"])
    async def health(request: Request) -> dict[str, str]:
        """Liveness probe for ALB / ECS health checks."""
        return {"status": "ok", "llm_mode": request.app.state.services.llm.mode}

    # Handlers are resolved along the exception MRO, so subclasses must be registered too.
    @app.exception_handler(LLMUnavailableError)
    async def _llm_unavailable(_: Request, exc: LLMUnavailableError) -> JSONResponse:
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))

    @app.exception_handler(LLMServiceError)
    async def _llm_failed(_: Request, exc: LLMServiceError) -> JSONResponse:
        # Messages are written by LLMService and contain no secrets (e.g. "credit balance is too low").
        return _error(status.HTTP_502_BAD_GATEWAY, f"The language model failed: {exc}")

    @app.exception_handler(ItemNotFoundError)
    async def _not_found(_: Request, exc: ItemNotFoundError) -> JSONResponse:
        return _error(status.HTTP_404_NOT_FOUND, str(exc))

    @app.exception_handler(DuplicateItemError)
    async def _duplicate(_: Request, exc: DuplicateItemError) -> JSONResponse:
        return _error(status.HTTP_409_CONFLICT, str(exc))

    @app.exception_handler(StorageError)
    async def _storage_failed(_: Request, exc: StorageError) -> JSONResponse:
        logger.error("Storage failure: %s", exc)
        return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "A storage error occurred.")

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
