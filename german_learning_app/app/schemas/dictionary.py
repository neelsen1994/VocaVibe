"""Strict Pydantic contracts for every structured LLM output.

These models double as the JSON Schemas handed to the LLM via
``with_structured_output``. To stay compatible with OpenAI *strict* structured
outputs they follow three rules:

* ``extra="forbid"`` -> ``additionalProperties: false``
* every field is required (optional values are expressed as ``X | None``
  without a default, so the model must emit an explicit ``null``)
* constraints the provider cannot enforce are validated in code instead
  (see ``app.agents.validation``).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Tense = Literal["present", "praeteritum", "perfekt"]
TENSES: tuple[Tense, ...] = ("present", "praeteritum", "perfekt")

PartOfSpeech = Literal[
    "verb",
    "noun",
    "adjective",
    "adverb",
    "preposition",
    "conjunction",
    "pronoun",
    "article",
    "interjection",
    "idiom",
    "phrase",
    "other",
]

IntentLabel = Literal["single_word", "phrase", "idiom", "invalid"]


class StrictModel(BaseModel):
    """Base model: rejects unknown keys and trims incidental whitespace."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Dictionary payload
# ---------------------------------------------------------------------------


class PersonForms(StrictModel):
    """One conjugated form per grammatical person (without the pronoun)."""

    ich: str = Field(description="1st person singular, e.g. 'trete ein'.")
    du: str = Field(description="2nd person singular, e.g. 'trittst ein'.")
    er_sie_es: str = Field(description="3rd person singular, e.g. 'tritt ein'.")
    wir: str = Field(description="1st person plural, e.g. 'treten ein'.")
    ihr: str = Field(description="2nd person plural, e.g. 'tretet ein'.")
    sie_Sie: str = Field(description="3rd person plural / formal 'Sie', e.g. 'treten ein'.")

    def all_forms(self) -> list[str]:
        return [self.ich, self.du, self.er_sie_es, self.wir, self.ihr, self.sie_Sie]


class ConjugationSet(StrictModel):
    """Core indicative conjugation table for a verb."""

    infinitive: str = Field(description="Infinitive, e.g. 'eintreten'.")
    auxiliary: Literal["haben", "sein", "haben_or_sein"] = Field(
        description="Perfekt auxiliary. Use 'haben_or_sein' when it depends on the sense."
    )
    partizip_ii: str = Field(description="Past participle, e.g. 'eingetreten'.")
    present: PersonForms = Field(description="Präsens forms.")
    praeteritum: PersonForms = Field(description="Präteritum forms.")
    perfekt: PersonForms = Field(description="Perfekt forms: auxiliary + Partizip II, e.g. 'bin eingetreten'.")

    def forms_for(self, tense: Tense) -> PersonForms:
        return {"present": self.present, "praeteritum": self.praeteritum, "perfekt": self.perfekt}[tense]


class SentenceExample(StrictModel):
    """A natural example sentence illustrating one context of the word."""

    german: str = Field(description="Complete, grammatical German sentence.")
    english: str = Field(description="Faithful English translation.")
    used_conjugation: str = Field(
        description=(
            "The inflected form of the headword exactly as it appears in the sentence. "
            "For separable verbs list all parts in order, e.g. 'tritt ein' or 'ist eingetreten'."
        )
    )
    tense: Tense | None = Field(description="Tense of used_conjugation for verbs; null for non-verbs.")


class TenseVariant(StrictModel):
    """One row of a context's conjugation matrix: the same sentence in one tense."""

    tense: Tense
    german: str = Field(description="The context's anchor sentence rendered in this tense.")
    english: str
    used_conjugation: str = Field(description="The verb form exactly as it appears, e.g. 'ist eingetreten'.")


class WordContext(StrictModel):
    """One distinct sense / usage context of the headword."""

    nuance_description: str = Field(
        description="What distinguishes this sense: meaning, register, typical collocations, governed case."
    )
    english_gloss: str = Field(description="Short English equivalent for this sense.")
    examples: list[SentenceExample] = Field(description="Two example sentences for this sense.")
    tense_matrix: list[TenseVariant] | None = Field(
        description=(
            "Verbs only: ONE anchor sentence for this sense shown in present, praeteritum and perfekt. "
            "Only the verb form may change between rows. Null for non-verbs."
        )
    )


class HomonymItem(StrictModel):
    """A distinct lexeme sharing the headword's spelling (e.g. die Bank: bench vs. bank)."""

    word: str
    article: Literal["der", "die", "das"] | None = Field(description="Definite article for nouns, else null.")
    part_of_speech: PartOfSpeech
    short_definition: str = Field(description="One-line English definition, including plural if it differs.")


class DictionaryResponse(StrictModel):
    """Full structured dictionary entry produced by the extraction agent."""

    word: str = Field(description="Canonical headword (infinitive / nominative singular).")
    part_of_speech: PartOfSpeech
    homonyms: list[HomonymItem] = Field(description="Distinct homonyms; empty list when none exist.")
    conjugations: ConjugationSet | None = Field(description="Required for verbs, null otherwise.")
    contexts: list[WordContext] = Field(description="Distinct senses, most frequent first.")


# ---------------------------------------------------------------------------
# Agent auxiliary outputs
# ---------------------------------------------------------------------------


class IntentAnalysis(StrictModel):
    """Output of the intent-analysis node."""

    is_valid: bool = Field(description="False when the input is not (recognisably) German.")
    normalized_input: str = Field(description="Input with spelling / speech-recognition errors corrected.")
    root_form: str = Field(description="Lemma: infinitive for verbs, nominative singular for nouns.")
    intent: IntentLabel
    is_idiom: bool
    candidate_words: list[str] = Field(
        description="Up to 5 German words the speaker most plausibly meant, best first. Empty if unambiguous."
    )
    rejection_reason: str | None = Field(description="Why the input was rejected; null when valid.")


class GrammarIssue(StrictModel):
    context_index: int = Field(description="Zero-based index into contexts.")
    example_index: int = Field(description="Zero-based index into that context's examples (or tense_matrix).")
    issue: str = Field(description="What is wrong (grammar, tense mismatch, mistranslation...).")
    suggested_fix: str = Field(description="Corrected sentence or field value.")


class GrammarCritique(StrictModel):
    """Output of the LLM grammar critic."""

    is_valid: bool
    issues: list[GrammarIssue]


class PracticeSentenceSet(StrictModel):
    """Fresh practice sentences generated for a single sense of a word."""

    sentences: list[SentenceExample]
