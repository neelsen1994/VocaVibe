"""End-to-end tests in offline mode (no network, no API key)."""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agents.graph import build_dictionary_graph
from app.agents.state import initial_state
from app.agents.validation import find_rule_violations
from app.config import Settings
from app.main import create_app
from app.schemas.dictionary import DictionaryResponse, GrammarCritique, IntentAnalysis
from app.services.fuzzy_service import cologne_phonetic, fuzzy_compare, levenshtein, rank_candidates
from app.services.lexicon import candidate_vocabulary, lookup_offline_entry
from app.services.llm_service import LLMService, LLMServiceError
from app.services.storage_service import (
    ConditionalCheckFailedError,
    MockDynamoDB,
    TransactionCanceledError,
    TransactWriteOp,
)

OFFLINE = Settings(openai_api_key=None)
USER = "learner-1"


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(OFFLINE)) as test_client:
        yield test_client


def _entry(client: TestClient, word: str = "eintreten") -> dict:
    response = client.post("/api/analyze", json={"text": word})
    assert response.status_code == 200, response.text
    return response.json()["result"]


def _context_by_gloss(entry: dict, gloss_prefix: str) -> dict:
    return next(c for c in entry["contexts"] if c["english_gloss"].startswith(gloss_prefix))


def _save(client: TestClient, entry: dict, context: dict) -> dict:
    response = client.post(
        "/api/flashcard/save",
        json={"user_id": USER, "word": entry["word"], "part_of_speech": entry["part_of_speech"],
              "context": context, "conjugations": entry["conjugations"]},
    )
    assert response.status_code == 201, response.text
    return response.json()


# -- /api/analyze -------------------------------------------------------------


def test_analyze_verb_is_validated(client: TestClient) -> None:
    body = client.post("/api/analyze", json={"text": "eintreten"}).json()
    assert body["status"] == "validated"
    assert body["iterations"] == 1
    assert body["result"]["conjugations"]["perfekt"]["ich"] == "bin eingetreten"
    assert len(body["result"]["contexts"]) == 5


def test_analyze_noun_returns_homonyms_without_matrix(client: TestClient) -> None:
    entry = _entry(client, "bank")
    assert entry["conjugations"] is None
    assert len(entry["homonyms"]) == 2
    assert all(c["tense_matrix"] is None for c in entry["contexts"])


def test_analyze_respects_requested_tense(client: TestClient) -> None:
    body = client.post("/api/analyze", json={"text": "eintreten", "requested_tense": "perfekt"}).json()
    tenses = {ex["tense"] for ctx in body["result"]["contexts"] for ex in ctx["examples"]}
    assert tenses == {"perfekt"}


def test_analyze_rejects_invalid_input(client: TestClient) -> None:
    assert client.post("/api/analyze", json={"text": "1234 !!"}).status_code == 422


def test_analyze_unknown_word_offline_is_503(client: TestClient) -> None:
    assert client.post("/api/analyze", json={"text": "Schmetterling"}).status_code == 503


# -- USP 2: multi-tense conjugation matrix --------------------------------------


def test_every_verb_sense_has_a_same_sentence_tense_matrix(client: TestClient) -> None:
    for context in _entry(client)["contexts"]:
        assert [row["tense"] for row in context["tense_matrix"]] == ["present", "praeteritum", "perfekt"]


def test_matrix_tracks_sense_specific_auxiliary(client: TestClient) -> None:
    kick_in = _context_by_gloss(_entry(client), "to kick in")
    perfekt = next(r for r in kick_in["tense_matrix"] if r["tense"] == "perfekt")
    assert perfekt["used_conjugation"] == "hat eingetreten"


def test_matrix_rejects_rows_that_change_the_context() -> None:
    entry = lookup_offline_entry("eintreten")
    assert entry is not None
    entry.contexts[0].tense_matrix[2].german = "Gestern ist der Ernstfall plötzlich eingetreten."
    assert any("changes the sentence context" in e for e in find_rule_violations(entry))


def test_matrix_rejects_wrong_auxiliary() -> None:
    entry = lookup_offline_entry("eintreten")
    assert entry is not None
    row = entry.contexts[0].tense_matrix[2]
    row.german, row.used_conjugation = "Er wird leise in das Zimmer eingetreten.", "wird eingetreten"
    assert any("not a valid perfekt form" in e for e in find_rule_violations(entry))


# -- USP 3: voice-to-confirmation fuzzy loop ----------------------------------


def test_levenshtein_and_cologne_phonetics() -> None:
    assert levenshtein("eintretten", "eintreten") == 1
    assert cologne_phonetic("eintreten") == cologne_phonetic("eintretten") == cologne_phonetic("aintreeten")


def test_candidates_rank_intended_word_first() -> None:
    candidates = rank_candidates("eintretten", candidate_vocabulary())
    assert candidates[0].word == "eintreten"
    assert candidates[0].phonetic_match is True
    assert candidates[0].correction == "drop 't'"
    assert len(candidates) > 1  # near-misses such as eintreffen / betreten are offered too


def test_misheard_voice_input_asks_for_confirmation(client: TestClient) -> None:
    body = client.post("/api/analyze", json={"text": "eintretten", "source": "voice"}).json()
    assert body["status"] == "needs_confirmation"
    assert body["result"] is None
    assert body["candidates"][0]["word"] == "eintreten"


def test_confirmed_candidate_runs_analysis(client: TestClient) -> None:
    picked = client.post("/api/analyze", json={"text": "eintretten", "source": "voice"}).json()["candidates"][0]
    body = client.post("/api/analyze", json={"text": picked["word"], "source": "voice", "confirmed": True}).json()
    assert body["status"] == "validated"
    assert body["result"]["word"] == "eintreten"


def test_exact_inflected_voice_input_skips_confirmation(client: TestClient) -> None:
    body = client.post("/api/analyze", json={"text": "tritt ein", "source": "voice"}).json()
    assert body["status"] == "validated"
    assert body["root_form"] == "eintreten"


def test_typed_input_is_never_gated(client: TestClient) -> None:
    assert client.post("/api/analyze", json={"text": "eintretten"}).json()["status"] == "validated"


# -- critique loop --------------------------------------------------------------


class _ScriptedLLM(LLMService):
    """Returns a broken draft for the first ``bad_attempts`` extractions."""

    def __init__(self, bad_attempts: int) -> None:
        super().__init__(OFFLINE)
        self.bad_attempts = bad_attempts
        self.feedback_seen: list[list[str] | None] = []

    def extract_dictionary(self, root_form, *, is_idiom=False, requested_tense=None, feedback=None) -> DictionaryResponse:
        self.feedback_seen.append(feedback)
        entry = lookup_offline_entry("eintreten")
        assert entry is not None
        if len(self.feedback_seen) <= self.bad_attempts:
            entry.contexts[0].examples[0].used_conjugation = "trat ein"  # wrong tense for the sentence
        return entry


def test_critique_loops_back_with_feedback() -> None:
    llm = _ScriptedLLM(bad_attempts=1)
    final = build_dictionary_graph(llm, max_retries=2).invoke(initial_state("eintreten"))
    assert final["iteration_count"] == 2
    assert final["critique_errors"] == []
    assert final["validated_json"]
    assert llm.feedback_seen[0] is None and llm.feedback_seen[1]


def test_critique_stops_after_retry_budget() -> None:
    final = build_dictionary_graph(_ScriptedLLM(bad_attempts=99), max_retries=2).invoke(initial_state("eintreten"))
    assert final["iteration_count"] == 3  # 1 initial + 2 retries
    assert final["critique_errors"]
    assert final["validated_json"] == {}


# -- USP 1: context-isolated flashcards ---------------------------------------


def test_each_sense_is_its_own_card_and_duplicates_conflict(client: TestClient) -> None:
    entry = _entry(client)
    occur = _save(client, entry, _context_by_gloss(entry, "to occur"))
    enter = _save(client, entry, _context_by_gloss(entry, "to enter"))
    assert occur["card_id"] != enter["card_id"]
    assert occur["sense"] == "to occur, to set in"

    duplicate = client.post(
        "/api/flashcard/save",
        json={"user_id": USER, "word": "eintreten", "part_of_speech": "verb",
              "context": _context_by_gloss(entry, "to occur"), "conjugations": entry["conjugations"]},
    )
    assert duplicate.status_code == 409


def test_practice_is_empty_without_cards(client: TestClient) -> None:
    body = client.get("/api/flashcard/practice", params={"user_id": "nobody"}).json()
    assert body == {"user_id": "nobody", "strategy": "empty", "exercises": []}


def test_mistake_targets_only_that_sense(client: TestClient) -> None:
    entry = _entry(client)
    occur = _save(client, entry, _context_by_gloss(entry, "to occur"))
    enter = _save(client, entry, _context_by_gloss(entry, "to enter"))
    bank = _entry(client, "bank")
    _save(client, bank, bank["contexts"][0])

    review = client.get("/api/flashcard/practice", params={"user_id": USER}).json()
    assert review["strategy"] == "review"
    assert {ex["card_id"] for ex in review["exercises"]} >= {occur["card_id"], enter["card_id"]}
    assert all("expected_answer" not in ex for ex in review["exercises"])  # scored server-side

    drill = next(ex for ex in review["exercises"] if ex["card_id"] == occur["card_id"])
    assert drill["exercise_type"] == "tense_transform"
    attempt = client.post(
        "/api/flashcard/attempt",
        json={"user_id": USER, "exercise_id": drill["exercise_id"], "transcript": "Der Ernstfall hat eingetreten"},
    ).json()
    assert attempt["is_correct"] is False
    assert attempt["target_form_found"] is False  # 'hat' instead of 'ist'

    targeted = client.get("/api/flashcard/practice", params={"user_id": USER, "limit": 10}).json()
    assert targeted["strategy"] == "error_targeted"
    card_ids = [ex["card_id"] for ex in targeted["exercises"]]
    assert card_ids[0] == occur["card_id"]
    assert enter["card_id"] not in card_ids  # sibling sense stays out of the targeted session
    assert {ex["sense"] for ex in targeted["exercises"] if ex["word"] == "eintreten"} == {"to occur, to set in"}
    # The fresh sentences focus on the tense the learner just got wrong.
    assert any(ex["exercise_type"] == "cloze" and ex["tense"] == drill["tense"] for ex in targeted["exercises"])


def test_cloze_accepts_gap_only_umlaut_free_answer(client: TestClient) -> None:
    entry = _entry(client)
    occur = _save(client, entry, _context_by_gloss(entry, "to occur"))
    review = client.get("/api/flashcard/practice", params={"user_id": USER}).json()
    drill = next(ex for ex in review["exercises"] if ex["card_id"] == occur["card_id"])
    wrong = client.post(
        "/api/flashcard/attempt",
        json={"user_id": USER, "exercise_id": drill["exercise_id"], "transcript": "Der Ernstfall hat eingetreten"},
    )
    assert wrong.status_code == 200

    targeted = client.get("/api/flashcard/practice", params={"user_id": USER}).json()
    cloze = next(ex for ex in targeted["exercises"] if ex["exercise_type"] == "cloze")
    answer = client.post(
        "/api/flashcard/attempt",
        json={"user_id": USER, "exercise_id": cloze["exercise_id"], "transcript": "ist eingetreten"},
    ).json()
    assert answer["is_correct"] is True, answer


def test_attempt_unknown_exercise_is_404(client: TestClient) -> None:
    response = client.post(
        "/api/flashcard/attempt", json={"user_id": USER, "exercise_id": "missing", "transcript": "hallo"}
    )
    assert response.status_code == 404


# -- Claude backend wiring (fake client, no network) ----------------------------


class _FakeClaude:
    """Stands in for ``anthropic.Anthropic``; records every ``beta.messages.parse`` call."""

    def __init__(self, outputs: dict, stop_reason: str = "end_turn") -> None:
        self.outputs, self.stop_reason, self.calls = outputs, stop_reason, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(category="cyber") if self.stop_reason == "refusal" else None,
            parsed_output=self.outputs.get(kwargs["output_format"]),
            model=kwargs["model"],
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )


def _claude_outputs() -> dict:
    return {
        IntentAnalysis: IntentAnalysis(
            is_valid=True, normalized_input="eintreten", root_form="eintreten", intent="single_word",
            is_idiom=False, candidate_words=[], rejection_reason=None,
        ),
        DictionaryResponse: lookup_offline_entry("eintreten"),
        GrammarCritique: GrammarCritique(is_valid=True, issues=[]),
    }


def test_claude_runs_the_full_graph_with_structured_outputs() -> None:
    fake = _FakeClaude(_claude_outputs())
    llm = LLMService(Settings(anthropic_model="claude-opus-5-5"), anthropic_client=fake)
    final = build_dictionary_graph(llm).invoke(initial_state("eintreten"))

    assert final["validated_json"]["word"] == "eintreten"
    assert [c["output_format"] for c in fake.calls] == [IntentAnalysis, DictionaryResponse, GrammarCritique]
    call = fake.calls[1]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "medium"}
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in call and "thinking" not in call
    assert llm.mode == "anthropic:claude-opus-5-5"


def test_claude_haiku_skips_server_side_fallback() -> None:
    fake = _FakeClaude(_claude_outputs())
    LLMService(Settings(anthropic_model="claude-haiku-5-5"), anthropic_client=fake).analyze_intent("eintreten")
    assert "fallbacks" not in fake.calls[0] and "betas" not in fake.calls[0]


def test_claude_refusal_becomes_service_error() -> None:
    llm = LLMService(Settings(), anthropic_client=_FakeClaude(_claude_outputs(), stop_reason="refusal"))
    with pytest.raises(LLMServiceError, match="declined"):
        llm.analyze_intent("eintreten")


def test_anthropic_key_selects_claude_provider() -> None:
    settings = Settings(anthropic_api_key="sk-ant-test", openai_api_key="sk-openai-test")
    assert settings.active_provider == "anthropic"
    assert LLMService(settings).mode == "anthropic:claude-opus-5-5"  # client construction makes no request
    assert Settings(llm_provider="openai", anthropic_api_key="x").active_provider == "openai"


# -- storage units --------------------------------------------------------------


def test_dynamodb_ttl_and_conditions() -> None:
    now = [1_000.0]
    db = MockDynamoDB(clock=lambda: now[0])
    db.create_table("t")
    db.put_item("t", {"PK": "U#1", "SK": "A", "expires_at": 1_500})
    with pytest.raises(ConditionalCheckFailedError):
        db.put_item("t", {"PK": "U#1", "SK": "A"}, condition="attribute_not_exists")
    now[0] = 2_000.0
    assert db.get_item("t", "U#1", "A") is None
    assert db.purge_expired() == 1


def test_dynamodb_transaction_is_atomic() -> None:
    db = MockDynamoDB()
    db.create_table("t")
    db.put_item("t", {"PK": "U#1", "SK": "taken"})
    with pytest.raises(TransactionCanceledError):
        db.transact_write([
            TransactWriteOp("put", "t", item={"PK": "U#1", "SK": "new"}),
            TransactWriteOp("put", "t", item={"PK": "U#1", "SK": "taken"}, condition="attribute_not_exists"),
        ])
    assert db.get_item("t", "U#1", "new") is None


def test_fuzzy_compare_folds_umlauts() -> None:
    assert fuzzy_compare("Die Tür ist groß.", "die tuer ist gross").score == 1.0
