"""HTTP request/response contracts for the public API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.schemas.dictionary import ConjugationSet, DictionaryResponse, IntentLabel, PartOfSpeech, Tense, WordContext

# Restricting the alphabet keeps ids safe to embed in DynamoDB keys ("USER#<id>").
# In production the user id comes from the Cognito JWT ``sub`` claim, not the body.
UserId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
EntityId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
InputSource = Literal["text", "voice"]
ExerciseType = Literal["cloze", "tense_transform", "translate_to_german"]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# /api/analyze (with the voice-to-confirmation loop)
# ---------------------------------------------------------------------------


class AnalyzeRequest(ApiModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    source: InputSource = Field(
        default="text", description="'voice' marks speech-to-text input and turns on the confirmation loop."
    )
    confirmed: bool = Field(
        default=False,
        description="Set when re-submitting a word the learner picked from `candidates`; skips confirmation.",
    )
    requested_tense: Tense | None = Field(
        default=None, description="If set, every example sentence must use this tense."
    )


class WordCandidateOut(ApiModel):
    word: str = Field(description="Lemma to re-submit with confirmed=true.")
    matched_form: str
    score: float
    edit_distance: int
    phonetic_match: bool = Field(description="Sounds identical (Kölner Phonetik); only the articulation differs.")
    correction: str = Field(description="How to get from what was heard to this word, e.g. \"drop 't'\".")


class AnalyzeResponse(ApiModel):
    status: Literal["validated", "unverified", "needs_confirmation"] = Field(
        description=(
            "'needs_confirmation': voice input was not an exact word, so pick from `candidates`. "
            "'unverified': the critic still had objections after the retry budget was spent."
        )
    )
    input: str
    intent: IntentLabel
    root_form: str
    is_idiom: bool
    iterations: int
    critique_errors: list[str]
    candidates: list[WordCandidateOut]
    llm_mode: str
    result: DictionaryResponse | None


# ---------------------------------------------------------------------------
# Flashcards & practice
# ---------------------------------------------------------------------------


class FlashcardSaveRequest(ApiModel):
    """One card = one sense. Save each context of a word as its own card."""

    user_id: UserId
    word: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    part_of_speech: PartOfSpeech
    context: WordContext
    conjugations: ConjugationSet | None = Field(
        default=None, description="The entry's conjugation table; enables tense drills for verbs."
    )
    ttl_days: int | None = Field(default=None, ge=1, le=3650, description="Overrides the default card TTL.")


class FlashcardSaveResponse(ApiModel):
    card_id: str
    user_id: str
    word: str
    sense: str
    created_at: str
    expires_at: int


class PracticeExercise(ApiModel):
    exercise_id: str
    card_id: str
    word: str
    sense: str = Field(description="The single meaning this exercise trains.")
    exercise_type: ExerciseType
    prompt: str
    hint: str
    tense: Tense | None
    focus_reason: str


class PracticeResponse(ApiModel):
    user_id: str
    strategy: Literal["error_targeted", "review", "empty"]
    exercises: list[PracticeExercise]


class FlashcardAttemptRequest(ApiModel):
    user_id: UserId
    exercise_id: EntityId
    transcript: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    source: InputSource = "voice"


class FlashcardAttemptResponse(ApiModel):
    is_correct: bool
    target_form_found: bool = Field(description="Whether the drilled conjugation appears in the answer.")
    score: float
    threshold: float
    expected: str
    target_form: str
    normalized_expected: str
    normalized_transcript: str
    mismatches: list[str]
