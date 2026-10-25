"""Answer scoring and RAG-driven, sense-isolated practice generation.

Every failed attempt becomes an "error memory": a history event in DynamoDB
plus an embedded summary in the vector store, tagged with the *card* (and so
the single sense) it belongs to. A practice request embeds the learner's
recent mistakes, retrieves similar error memories, and ranks the senses they
point to. Only those senses are drilled, with freshly generated sentences.
Sibling senses of the same word are deliberately kept out of the session: a
learner struggling with "eintreten = to occur" is not shown "eintreten = to
enter".

Exercises are persisted with a short TTL and scored server-side, so expected
answers never reach the client.
"""

import logging
import re
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from app.agents.validation import sentence_violations
from app.schemas.api import FlashcardAttemptResponse, PracticeExercise, PracticeResponse
from app.schemas.dictionary import TENSES, ConjugationSet, SentenceExample, Tense, TenseVariant, WordContext
from app.services.fuzzy_service import contains_form, fuzzy_compare
from app.services.llm_service import LLMService, LLMServiceError
from app.services.storage_service import FlashcardRepository, MockVectorStore, StorageError

logger = logging.getLogger(__name__)

ERROR_EVENT = "PRACTICE_ERROR"
SUCCESS_EVENT = "PRACTICE_SUCCESS"
BLANK = "____"
TENSE_LABELS: dict[Tense, str] = {"present": "Präsens", "praeteritum": "Präteritum", "perfekt": "Perfekt"}


def make_cloze(sentence: str, used_conjugation: str) -> str | None:
    """Blank out every part of ``used_conjugation``; None if any part is missing from the sentence."""
    result = sentence
    for part in used_conjugation.split():
        pattern = re.compile(rf"(?<!\w){re.escape(part)}(?!\w)", re.IGNORECASE)
        result, count = pattern.subn(BLANK, result, count=1)
        if count == 0:
            return None
    return result


class PracticeService:
    # How many recent mistakes form the retrieval query.
    RECENT_ERROR_WINDOW = 10
    # Fresh sentences generated per targeted sense.
    SENTENCES_PER_SENSE = 2

    def __init__(
        self,
        repository: FlashcardRepository,
        vectors: MockVectorStore,
        llm: LLMService,
        *,
        fuzzy_threshold: float,
    ) -> None:
        self._repo = repository
        self._vectors = vectors
        self._llm = llm
        self._threshold = fuzzy_threshold

    # -- practice generation ------------------------------------------------

    def build_practice_set(self, *, user_id: str, limit: int) -> PracticeResponse:
        cards = {c["card_id"]: c for c in self._repo.list_flashcards(user_id)}
        if not cards:
            return PracticeResponse(user_id=user_id, strategy="empty", exercises=[])

        errors = self._repo.list_history(user_id, event_type=ERROR_EVENT, limit=self.RECENT_ERROR_WINDOW)
        if errors:
            strategy = "error_targeted"
            exercises = self._error_targeted(user_id, cards, errors, limit)
        else:
            strategy = "review"
            exercises = self._review(cards, limit)

        self._repo.save_exercises(user_id, exercises)
        return PracticeResponse(
            user_id=user_id,
            strategy=strategy,
            exercises=[PracticeExercise.model_validate({k: ex[k] for k in PracticeExercise.model_fields}) for ex in exercises],
        )

    def _error_targeted(
        self, user_id: str, cards: Mapping[str, dict[str, Any]], errors: Sequence[Mapping[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        query = " ".join(e["payload"]["summary"] for e in errors)
        # Only error memories are searched, so every hit points at the exact sense the learner got wrong.
        hits = self._vectors.similarity_search(
            query, k=max(limit * 3, 10), metadata_filter={"user_id": user_id, "kind": "error"}
        )
        weight: dict[str, float] = defaultdict(float)
        focus: dict[str, list[Tense]] = defaultdict(list)
        for hit in hits:
            card_id = hit.metadata.get("card_id", "")
            if card_id not in cards:  # card expired via TTL
                continue
            weight[card_id] += max(hit.score, 0.0)
            tense = hit.metadata.get("tense")
            if tense in TENSES and tense not in focus[card_id]:
                focus[card_id].append(tense)

        exercises: list[dict[str, Any]] = []
        targeted_words: set[str] = set()
        for card_id in sorted(weight, key=weight.__getitem__, reverse=True):
            card = cards[card_id]
            targeted_words.add(card["word"].casefold())
            reason = f"You recently struggled with '{card['word']}' meaning '{card['sense']}'."
            exercises.extend(self._targeted_exercises(card, focus[card_id], reason))
            if len(exercises) >= limit:
                return exercises[:limit]

        # Top up with review of *other* words; sibling senses stay out of a targeted session.
        others = {cid: c for cid, c in cards.items() if c["word"].casefold() not in targeted_words}
        exercises.extend(self._review(others, limit - len(exercises)))
        return exercises[:limit]

    def _targeted_exercises(self, card: Mapping[str, Any], focus_tenses: list[Tense], reason: str) -> list[dict[str, Any]]:
        context = WordContext.model_validate(card["context"])
        conjugations = ConjugationSet.model_validate(card["conjugations"]) if card.get("conjugations") else None

        try:
            sentences = self._llm.generate_practice_sentences(
                word=card["word"],
                context=context,
                conjugations=conjugations,
                focus_tenses=focus_tenses,
                count=self.SENTENCES_PER_SENSE,
            )
        except LLMServiceError:
            logger.exception("Practice generation failed for card %s; using stored sentences", card["card_id"])
            sentences = list(context.examples)
        # Generated text gets the same deterministic checks as dictionary entries.
        sentences = [s for s in sentences if not sentence_violations(s, conjugations, "practice")]

        exercises = [self._cloze(card, s, reason) for s in sentences]
        if context.tense_matrix:
            target = focus_tenses[0] if focus_tenses else "perfekt"
            exercises.append(self._tense_transform(card, context.tense_matrix, target, reason))
        return [ex for ex in exercises if ex is not None]

    def _review(self, cards: Mapping[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """One exercise per card from stored content; no model call needed."""
        exercises: list[dict[str, Any]] = []
        for position, card in enumerate(cards.values()):
            if len(exercises) >= limit:
                break
            context = WordContext.model_validate(card["context"])
            reason = "Spaced review of a saved flashcard."
            exercise = None
            if context.tense_matrix:
                # Rotate the target tense across cards so reviews cover the whole matrix.
                target = ("perfekt", "praeteritum", "present")[position % 3]
                exercise = self._tense_transform(card, context.tense_matrix, target, reason)
            elif context.examples:
                exercise = self._cloze(card, context.examples[0], reason)
            if exercise is not None:
                exercises.append(exercise)
        return exercises

    # -- exercise builders --------------------------------------------------

    @staticmethod
    def _base(card: Mapping[str, Any], reason: str) -> dict[str, Any]:
        return {
            "exercise_id": uuid.uuid4().hex,
            "card_id": card["card_id"],
            "word": card["word"],
            "sense": card["sense"],
            "part_of_speech": card["part_of_speech"],
            "nuance": card["context"]["nuance_description"],
            "focus_reason": reason,
        }

    def _cloze(self, card: Mapping[str, Any], sentence: SentenceExample, reason: str) -> dict[str, Any]:
        cloze = make_cloze(sentence.german, sentence.used_conjugation)
        label = f" ({TENSE_LABELS[sentence.tense]})" if sentence.tense else ""
        if cloze is None:
            return {
                **self._base(card, reason),
                "exercise_type": "translate_to_german",
                "prompt": f"Say it in German: {sentence.english}",
                "hint": f"{card['word']} = {card['sense']}",
                "tense": sentence.tense,
                "expected_answer": sentence.german,
                "target_form": sentence.used_conjugation,
            }
        return {
            **self._base(card, reason),
            "exercise_type": "cloze",
            "prompt": f"Fill the gap{label}: {cloze}",
            "hint": f"{sentence.english} | {card['word']} = {card['sense']}",
            "tense": sentence.tense,
            "expected_answer": sentence.german,
            "target_form": sentence.used_conjugation,
        }

    def _tense_transform(
        self, card: Mapping[str, Any], matrix: Sequence[TenseVariant], target_tense: Tense, reason: str
    ) -> dict[str, Any] | None:
        """Same sentence, same meaning, different tense: the learner moves the verb through the matrix."""
        by_tense = {v.tense: v for v in matrix}
        target = by_tense.get(target_tense)
        source = by_tense.get("present") if target_tense != "present" else by_tense.get("perfekt")
        if target is None or source is None:
            return None
        return {
            **self._base(card, reason),
            "exercise_type": "tense_transform",
            "prompt": f"Say this in the {TENSE_LABELS[target_tense]}: {source.german}",
            "hint": f"{target.english} | {card['word']} = {card['sense']}",
            "tense": target_tense,
            "expected_answer": target.german,
            "target_form": target.used_conjugation,
        }

    # -- attempts -----------------------------------------------------------

    def record_attempt(self, *, user_id: str, exercise_id: str, transcript: str) -> FlashcardAttemptResponse:
        exercise = self._repo.get_exercise(user_id, exercise_id)
        expected, target_form = exercise["expected_answer"], exercise["target_form"]

        # For a cloze the learner may say only the missing words or the whole sentence.
        is_fill_only = exercise["exercise_type"] == "cloze" and len(transcript.split()) <= len(target_form.split()) + 1
        match = fuzzy_compare(target_form if is_fill_only else expected, transcript)
        # Overall similarity alone would accept one wrong word in a long sentence,
        # so the drilled form itself must also be present.
        target_form_found = contains_form(transcript, target_form)
        is_correct = target_form_found and match.score >= self._threshold

        tense = exercise.get("tense")
        summary = (
            f"Mistake with '{exercise['word']}' meaning '{exercise['sense']}' "
            f"({tense or exercise['part_of_speech']}, form '{target_form}'): {exercise['nuance']} "
            f"Expected \"{expected}\" but answered \"{transcript}\". " + "; ".join(match.mismatches)
        )
        event = self._repo.log_event(
            user_id,
            SUCCESS_EVENT if is_correct else ERROR_EVENT,
            {
                "exercise_id": exercise_id,
                "card_id": exercise["card_id"],
                "word": exercise["word"],
                "sense": exercise["sense"],
                "tense": tense,
                "expected": expected,
                "transcript": transcript,
                "score": match.score,
                "mismatches": list(match.mismatches),
                "summary": summary,
            },
        )
        if not is_correct:
            try:
                self._vectors.upsert(
                    [f"error#{event['event_id']}"],
                    [summary],
                    [{"user_id": user_id, "kind": "error", "card_id": exercise["card_id"], "tense": tense}],
                )
            except StorageError:
                # The history event is the source of truth; the index can be rebuilt from it.
                logger.exception("Failed to index error memory %s", event["event_id"])

        return FlashcardAttemptResponse(
            is_correct=is_correct,
            target_form_found=target_form_found,
            score=match.score,
            threshold=self._threshold,
            expected=expected,
            target_form=target_form,
            normalized_expected=match.normalized_expected,
            normalized_transcript=match.normalized_actual,
            mismatches=list(match.mismatches),
        )
