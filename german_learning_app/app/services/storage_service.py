"""In-memory stand-ins for the persistence tier.

* ``MockDynamoDB``      -> Amazon DynamoDB (single-table design, TTL, TransactWriteItems)
* ``FlashcardRepository`` -> data-access layer that will keep its interface when
  swapped onto boto3
* ``MockVectorStore``   -> Amazon OpenSearch Serverless vector collection (k-NN)

All classes are thread-safe: FastAPI runs sync work in a thread pool, so
concurrent requests can hit the same instance. Values are deep-copied on the
way in and out, so callers can never mutate stored state by reference, which
is the same isolation a network-backed store gives.
"""

import copy
import hashlib
import logging
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import numpy as np
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

Item = dict[str, Any]
Key = tuple[str, str]
Condition = Literal["attribute_not_exists", "attribute_exists"]

PK = "PK"
SK = "SK"
# DynamoDB TTL sweeps items whose epoch-seconds attribute is in the past.
TTL_ATTRIBUTE = "expires_at"
# Hard limit of TransactWriteItems.
MAX_TRANSACTION_ITEMS = 100


# ---------------------------------------------------------------------------
# Errors (names mirror the botocore error codes they will map to)
# ---------------------------------------------------------------------------


class StorageError(Exception):
    """Base class for persistence failures."""


class TableNotFoundError(StorageError):
    """ResourceNotFoundException."""


class StorageValidationError(StorageError):
    """ValidationException: malformed request (missing keys, bad limits...)."""


class ConditionalCheckFailedError(StorageError):
    """ConditionalCheckFailedException."""


class TransactionCanceledError(StorageError):
    """TransactionCanceledException: carries one reason per operation."""

    def __init__(self, reasons: list[str | None]) -> None:
        self.reasons = reasons
        details = "; ".join(f"op[{i}]: {r}" for i, r in enumerate(reasons) if r)
        super().__init__(f"Transaction cancelled ({details})")


class ItemNotFoundError(StorageError):
    """Raised by repositories when a requested entity does not exist."""


class DuplicateItemError(StorageError):
    """Raised by repositories when an entity with the same identity already exists."""


def _extract_key(item: Mapping[str, Any]) -> Key:
    pk, sk = item.get(PK), item.get(SK)
    if not isinstance(pk, str) or not pk or not isinstance(sk, str) or not sk:
        raise StorageValidationError("Items require non-empty string 'PK' and 'SK' attributes")
    return pk, sk


# ---------------------------------------------------------------------------
# MockDynamoDB
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransactWriteOp:
    """One entry of a TransactWriteItems call."""

    action: Literal["put", "delete"]
    table: str
    item: Item | None = None  # required for "put"
    key: Key | None = None  # required for "delete"
    condition: Condition | None = None

    def target_key(self) -> Key:
        if self.action == "put":
            if self.item is None:
                raise StorageValidationError("A 'put' operation requires an item")
            return _extract_key(self.item)
        if self.key is None:
            raise StorageValidationError("A 'delete' operation requires a key")
        return self.key


class MockDynamoDB:
    """In-memory DynamoDB emulation with partition/sort keys, TTL and transactions.

    Storage layout is ``table -> partition key -> sort key -> item``, so a query
    touches a single partition exactly like the real service.

    TTL note: real DynamoDB deletes expired items lazily (up to ~48h later) and
    may still return them; this mock hides them immediately, which is the
    behaviour application code should emulate with a filter in production.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._tables: dict[str, dict[str, dict[str, Item]]] = {}
        self._lock = threading.RLock()

    # -- control plane ------------------------------------------------------

    def create_table(self, name: str) -> None:
        """Idempotently create a table."""
        with self._lock:
            self._tables.setdefault(name, {})

    # -- data plane ---------------------------------------------------------

    def put_item(self, table: str, item: Mapping[str, Any], *, condition: Condition | None = None) -> None:
        pk, sk = _extract_key(item)
        stored = copy.deepcopy(dict(item))
        with self._lock:
            partitions = self._get_table(table)
            self._check_condition(partitions, (pk, sk), condition)
            partitions.setdefault(pk, {})[sk] = stored

    def get_item(self, table: str, pk: str, sk: str) -> Item | None:
        with self._lock:
            item = self._live_item(self._get_table(table), (pk, sk), self._clock())
            return copy.deepcopy(item) if item is not None else None

    def delete_item(self, table: str, pk: str, sk: str, *, condition: Condition | None = None) -> bool:
        """Delete an item; returns whether a live item was removed."""
        with self._lock:
            partitions = self._get_table(table)
            self._check_condition(partitions, (pk, sk), condition)
            existed = self._live_item(partitions, (pk, sk), self._clock()) is not None
            partitions.get(pk, {}).pop(sk, None)
            return existed

    def query(
        self,
        table: str,
        pk: str,
        *,
        sk_prefix: str | None = None,
        scan_forward: bool = True,
        limit: int | None = None,
    ) -> list[Item]:
        """KeyConditionExpression ``PK = :pk AND begins_with(SK, :prefix)``."""
        if limit is not None and limit < 1:
            raise StorageValidationError("limit must be a positive integer")
        with self._lock:
            now = self._clock()
            partition = self._get_table(table).get(pk, {})
            matches = [
                item
                for sk, item in partition.items()
                if (sk_prefix is None or sk.startswith(sk_prefix)) and not self._is_expired(item, now)
            ]
            matches.sort(key=lambda it: it[SK], reverse=not scan_forward)
            if limit is not None:
                matches = matches[:limit]
            return copy.deepcopy(matches)

    def transact_write(self, operations: Sequence[TransactWriteOp]) -> None:
        """All-or-nothing write: every condition is checked before anything is applied."""
        if not operations:
            raise StorageValidationError("A transaction requires at least one operation")
        if len(operations) > MAX_TRANSACTION_ITEMS:
            raise StorageValidationError(f"A transaction supports at most {MAX_TRANSACTION_ITEMS} operations")

        targets = [(op.table, op.target_key()) for op in operations]
        if len(set(targets)) != len(targets):
            raise StorageValidationError("A transaction cannot include multiple operations on the same item")
        staged = [copy.deepcopy(op.item) if op.item is not None else None for op in operations]

        with self._lock:
            reasons: list[str | None] = []
            for op, (table, key) in zip(operations, targets, strict=True):
                try:
                    self._check_condition(self._get_table(table), key, op.condition)
                    reasons.append(None)
                except ConditionalCheckFailedError as exc:
                    reasons.append(str(exc))
            if any(reasons):
                raise TransactionCanceledError(reasons)

            for op, (table, (pk, sk)), item in zip(operations, targets, staged, strict=True):
                partitions = self._tables[table]
                if op.action == "put":
                    partitions.setdefault(pk, {})[sk] = item  # type: ignore[assignment]
                else:
                    partitions.get(pk, {}).pop(sk, None)

    def purge_expired(self) -> int:
        """Physically remove expired items (the background TTL sweeper). Returns the count."""
        removed = 0
        with self._lock:
            now = self._clock()
            for partitions in self._tables.values():
                for pk in list(partitions):
                    partition = partitions[pk]
                    for sk in [s for s, it in partition.items() if self._is_expired(it, now)]:
                        del partition[sk]
                        removed += 1
                    if not partition:
                        del partitions[pk]
        return removed

    # -- internals ----------------------------------------------------------

    def _get_table(self, name: str) -> dict[str, dict[str, Item]]:
        try:
            return self._tables[name]
        except KeyError as exc:
            raise TableNotFoundError(f"Table {name!r} does not exist") from exc

    @staticmethod
    def _is_expired(item: Mapping[str, Any], now: float) -> bool:
        expires_at = item.get(TTL_ATTRIBUTE)
        return isinstance(expires_at, int | float) and expires_at <= now

    def _live_item(self, partitions: dict[str, dict[str, Item]], key: Key, now: float) -> Item | None:
        item = partitions.get(key[0], {}).get(key[1])
        if item is None or self._is_expired(item, now):
            return None
        return item

    def _check_condition(self, partitions: dict[str, dict[str, Item]], key: Key, condition: Condition | None) -> None:
        if condition is None:
            return
        exists = self._live_item(partitions, key, self._clock()) is not None
        if condition == "attribute_not_exists" and exists:
            raise ConditionalCheckFailedError(f"Item {key} already exists")
        if condition == "attribute_exists" and not exists:
            raise ConditionalCheckFailedError(f"Item {key} does not exist")


# ---------------------------------------------------------------------------
# Repository (single-table design)
# ---------------------------------------------------------------------------


class FlashcardRepository:
    """Flashcards, practice exercises and the learner's activity log in one DynamoDB table.

    A flashcard is one *sense* of a word, never the whole word: "eintreten (to
    occur)" and "eintreten (to enter)" are separate items with separate
    histories, so practice can target one meaning in isolation.

    Access patterns:

    ================================  ==============  =========================
    Pattern                           PK              SK
    ================================  ==============  =========================
    Get / list a user's flashcards    USER#<user_id>  CARD#<card_id>
    Get a generated exercise          USER#<user_id>  EXERCISE#<exercise_id>
    Recent history (newest first)     USER#<user_id>  HISTORY#<iso-ts>#<event>
    ================================  ==============  =========================

    ISO-8601 UTC timestamps sort lexicographically, so ``scan_forward=False``
    yields the newest events first without a secondary index.
    """

    CARD_PREFIX = "CARD#"
    EXERCISE_PREFIX = "EXERCISE#"
    HISTORY_PREFIX = "HISTORY#"

    def __init__(
        self,
        db: MockDynamoDB,
        *,
        table_name: str = "vocavibe-main",
        flashcard_ttl_days: int = 180,
        history_ttl_days: int = 365,
        exercise_ttl_hours: int = 24,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._db = db
        self._table = table_name
        self._flashcard_ttl_days = flashcard_ttl_days
        self._history_ttl_days = history_ttl_days
        self._exercise_ttl_hours = exercise_ttl_hours
        self._now = now
        db.create_table(table_name)

    @staticmethod
    def user_pk(user_id: str) -> str:
        return f"USER#{user_id}"

    @staticmethod
    def sense_card_id(word: str, english_gloss: str) -> str:
        """Deterministic id per (word, sense): re-saving the same sense is detected, not duplicated."""
        identity = f"{word.strip().casefold()}|{english_gloss.strip().casefold()}"
        return hashlib.sha256(identity.encode()).hexdigest()[:32]

    def save_flashcard(
        self,
        *,
        user_id: str,
        word: str,
        part_of_speech: str,
        context: Mapping[str, Any],
        conjugations: Mapping[str, Any] | None = None,
        ttl_days: int | None = None,
    ) -> Item:
        """Atomically persist a single-sense flashcard and its CARD_SAVED audit event.

        Raises ``DuplicateItemError`` if this user already saved the same sense.
        """
        now = self._now()
        card_id = self.sense_card_id(word, context["english_gloss"])
        card: Item = {
            PK: self.user_pk(user_id),
            SK: f"{self.CARD_PREFIX}{card_id}",
            "entity": "FLASHCARD",
            "card_id": card_id,
            "user_id": user_id,
            "word": word,
            "sense": context["english_gloss"],
            "part_of_speech": part_of_speech,
            "context": dict(context),
            "conjugations": dict(conjugations) if conjugations is not None else None,
            "created_at": now.isoformat(),
            TTL_ATTRIBUTE: int((now + timedelta(days=ttl_days or self._flashcard_ttl_days)).timestamp()),
        }
        event = self._build_event(user_id, "CARD_SAVED", {"card_id": card_id, "word": word}, now)
        try:
            self._db.transact_write(
                [
                    TransactWriteOp("put", self._table, item=card, condition="attribute_not_exists"),
                    TransactWriteOp("put", self._table, item=event, condition="attribute_not_exists"),
                ]
            )
        except TransactionCanceledError as exc:
            if exc.reasons and exc.reasons[0]:
                raise DuplicateItemError(
                    f"'{word}' in the sense '{context['english_gloss']}' is already saved"
                ) from exc
            raise
        return copy.deepcopy(card)

    def save_exercises(self, user_id: str, exercises: Sequence[Mapping[str, Any]]) -> None:
        """Persist a generated practice set (short TTL) so attempts are scored server-side."""
        if not exercises:
            return
        expires_at = int((self._now() + timedelta(hours=self._exercise_ttl_hours)).timestamp())
        self._db.transact_write(
            [
                TransactWriteOp(
                    "put",
                    self._table,
                    item={
                        **ex,
                        PK: self.user_pk(user_id),
                        SK: f"{self.EXERCISE_PREFIX}{ex['exercise_id']}",
                        "entity": "EXERCISE",
                        TTL_ATTRIBUTE: expires_at,
                    },
                )
                for ex in exercises
            ]
        )

    def get_exercise(self, user_id: str, exercise_id: str) -> Item:
        item = self._db.get_item(self._table, self.user_pk(user_id), f"{self.EXERCISE_PREFIX}{exercise_id}")
        if item is None:
            raise ItemNotFoundError(f"Exercise {exercise_id!r} not found or expired")
        return item

    def get_flashcard(self, user_id: str, card_id: str) -> Item:
        item = self._db.get_item(self._table, self.user_pk(user_id), f"{self.CARD_PREFIX}{card_id}")
        if item is None:
            raise ItemNotFoundError(f"Flashcard {card_id!r} not found for user {user_id!r}")
        return item

    def list_flashcards(self, user_id: str) -> list[Item]:
        """All live flashcards, newest first."""
        cards = self._db.query(self._table, self.user_pk(user_id), sk_prefix=self.CARD_PREFIX)
        return sorted(cards, key=lambda c: c["created_at"], reverse=True)

    def log_event(self, user_id: str, event_type: str, payload: Mapping[str, Any]) -> Item:
        event = self._build_event(user_id, event_type, payload, self._now())
        self._db.put_item(self._table, event, condition="attribute_not_exists")
        return event

    def list_history(self, user_id: str, *, event_type: str | None = None, limit: int = 50) -> list[Item]:
        """Newest-first history. ``event_type`` behaves like a FilterExpression.

        Unlike this mock, real DynamoDB applies ``Limit`` *before* filtering, so
        the production implementation must paginate until ``limit`` matches.
        """
        events = self._db.query(self._table, self.user_pk(user_id), sk_prefix=self.HISTORY_PREFIX, scan_forward=False)
        if event_type is not None:
            events = [e for e in events if e.get("event_type") == event_type]
        return events[:limit]

    def _build_event(self, user_id: str, event_type: str, payload: Mapping[str, Any], now: datetime) -> Item:
        event_id = uuid.uuid4().hex
        return {
            PK: self.user_pk(user_id),
            SK: f"{self.HISTORY_PREFIX}{now.isoformat(timespec='microseconds')}#{event_id}",
            "entity": "HISTORY",
            "event_id": event_id,
            "event_type": event_type,
            "user_id": user_id,
            "payload": copy.deepcopy(dict(payload)),
            "created_at": now.isoformat(),
            TTL_ATTRIBUTE: int((now + timedelta(days=self._history_ttl_days)).timestamp()),
        }


# ---------------------------------------------------------------------------
# MockVectorStore
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity in [-1, 1]


class MockVectorStore:
    """Cosine-similarity k-NN over an in-memory numpy matrix.

    Rows are L2-normalised on insert, so a search is a single matrix-vector
    product. Embeddings are computed *outside* the lock because a real embedder
    makes network calls; only the matrix mutation is serialised.
    """

    def __init__(self, embedder: Embeddings) -> None:
        self._embedder = embedder
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._positions: dict[str, int] = {}
        self._texts: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._matrix: np.ndarray | None = None  # shape (n, dim), float32

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)

    def upsert(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[str]:
        """Insert or replace documents by id."""
        metadatas = metadatas if metadatas is not None else [{}] * len(texts)
        if not (len(ids) == len(texts) == len(metadatas)):
            raise StorageValidationError("ids, texts and metadatas must have equal lengths")
        if not texts:
            return []
        if len(set(ids)) != len(ids):
            raise StorageValidationError("ids must be unique within one upsert call")
        if any(not t or not t.strip() for t in texts):
            raise StorageValidationError("Cannot index empty text")

        vectors = self._embed_documents(texts)

        with self._lock:
            if self._matrix is not None and vectors.shape[1] != self._matrix.shape[1]:
                raise StorageValidationError(
                    f"Embedding dimension {vectors.shape[1]} does not match index dimension {self._matrix.shape[1]}"
                )
            new_rows: list[np.ndarray] = []
            for doc_id, text, meta, vector in zip(ids, texts, metadatas, vectors, strict=True):
                position = self._positions.get(doc_id)
                if position is not None and self._matrix is not None:
                    self._matrix[position] = vector
                    self._texts[position] = text
                    self._metadata[position] = copy.deepcopy(dict(meta))
                    continue
                self._positions[doc_id] = len(self._ids)
                self._ids.append(doc_id)
                self._texts.append(text)
                self._metadata.append(copy.deepcopy(dict(meta)))
                new_rows.append(vector)
            if new_rows:
                stacked = np.vstack(new_rows)
                self._matrix = stacked if self._matrix is None else np.vstack([self._matrix, stacked])
        return list(ids)

    def add_texts(self, texts: Sequence[str], metadatas: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
        """Insert documents under freshly generated ids."""
        return self.upsert([uuid.uuid4().hex for _ in texts], texts, metadatas)

    def delete(self, ids: Sequence[str]) -> int:
        """Remove documents by id; returns how many were removed."""
        with self._lock:
            doomed = {self._positions[i] for i in ids if i in self._positions}
            if not doomed or self._matrix is None:
                return 0
            keep = [p for p in range(len(self._ids)) if p not in doomed]
            self._ids = [self._ids[p] for p in keep]
            self._texts = [self._texts[p] for p in keep]
            self._metadata = [self._metadata[p] for p in keep]
            self._matrix = self._matrix[keep] if keep else None
            self._positions = {doc_id: pos for pos, doc_id in enumerate(self._ids)}
            return len(doomed)

    def similarity_search(
        self,
        query: str,
        *,
        k: int = 4,
        metadata_filter: Mapping[str, Any] | None = None,
        min_score: float = -1.0,
    ) -> list[VectorSearchResult]:
        """Top-k documents by cosine similarity, optionally pre-filtered on exact metadata matches."""
        if k < 1:
            raise StorageValidationError("k must be a positive integer")
        if not query.strip():
            return []
        query_vector = self._normalise(np.asarray([self._embedder.embed_query(query)], dtype=np.float32))[0]

        with self._lock:
            if self._matrix is None:
                return []
            if query_vector.shape[0] != self._matrix.shape[1]:
                raise StorageValidationError("Query embedding dimension does not match the index")
            candidates = np.array(
                [
                    pos
                    for pos, meta in enumerate(self._metadata)
                    if not metadata_filter or all(meta.get(f) == v for f, v in metadata_filter.items())
                ],
                dtype=np.int64,
            )
            if candidates.size == 0:
                return []
            scores = self._matrix[candidates] @ query_vector
            order = np.argsort(-scores, kind="stable")[:k]
            return [
                VectorSearchResult(
                    id=self._ids[candidates[i]],
                    text=self._texts[candidates[i]],
                    metadata=copy.deepcopy(self._metadata[candidates[i]]),
                    score=float(scores[i]),
                )
                for i in order
                if float(scores[i]) >= min_score
            ]

    def _embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self._embedder.embed_documents(list(texts)), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise StorageError("Embedder returned an unexpected shape")
        return self._normalise(vectors)

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
