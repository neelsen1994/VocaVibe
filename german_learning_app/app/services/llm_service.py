"""LLM gateway, embeddings and agent tools.

``LLMService`` is the only module that talks to a model provider. Backends:

* **anthropic**: Claude through the official ``anthropic`` SDK, using
  ``beta.messages.parse`` structured outputs (the response is decoded against
  the Pydantic schema) plus server-side refusal fallbacks. Enabled when
  ``ANTHROPIC_API_KEY`` is set.
* **openai**: ``ChatOpenAI`` with OpenAI *strict* structured outputs, enabled
  when ``OPENAI_API_KEY`` is set (or when a LangChain chat model is injected).
* **offline**: deterministic heuristics plus the curated lexicon in
  ``app.services.lexicon``, so the API, the agent graph and the tests all run
  with no network access.
"""

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, TypeVar

import anthropic
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.schemas.dictionary import (
    ConjugationSet,
    DictionaryResponse,
    GrammarCritique,
    IntentAnalysis,
    PracticeSentenceSet,
    SentenceExample,
    Tense,
    WordContext,
)
from app.services.fuzzy_service import WordCandidate, fuzzy_compare, normalize_german, rank_candidates
from app.services.lexicon import (
    INFLECTION_INDEX,
    OFFLINE_IDIOMS,
    OFFLINE_LEXICON,
    candidate_vocabulary,
    lookup_offline_entry,
)

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_VALID_INPUT_RE = re.compile(r"^[A-Za-zÄÖÜäöüß' -]+$")
_EDGE_PUNCTUATION = ".,!?;:\"'„“”‚‘’«»()"

# Non-streaming ceiling: generous enough for a five-sense entry plus thinking,
# small enough to stay well inside the SDK's HTTP timeout.
CLAUDE_MAX_TOKENS = 16000
# On a safety-classifier decline the API re-runs the request on Anthropic's
# recommended fallback model inside the same call. Haiku has no fallback.
CLAUDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMServiceError(RuntimeError):
    """The model provider failed or returned output that violates the schema."""


class LLMUnavailableError(LLMServiceError):
    """The request needs a live model but the service runs in offline mode."""


# ---------------------------------------------------------------------------
# LangChain tools (ready for ``chat_model.bind_tools(AGENT_TOOLS)``)
# ---------------------------------------------------------------------------


@tool
def score_spoken_answer(expected: str, transcript: str) -> dict[str, Any]:
    """Score a speech-to-text transcript against the expected German sentence.

    Returns a 0..1 similarity score plus word-level mismatches. Umlauts and their
    digraph spellings (ä/ae, ß/ss) are treated as equal.
    """
    result = fuzzy_compare(expected, transcript)
    return {"score": result.score, "mismatches": list(result.mismatches)}


@tool
def suggest_intended_words(heard: str) -> list[dict[str, Any]]:
    """Rank the German words a learner most plausibly meant, by Levenshtein distance and Kölner Phonetik."""
    return [
        {"word": c.word, "score": c.score, "edit_distance": c.edit_distance, "correction": c.correction}
        for c in rank_candidates(heard, candidate_vocabulary())
    ]


@tool
def lookup_offline_lexicon(word: str) -> dict[str, Any] | None:
    """Return the curated dictionary entry for a German lemma, or null if it is not curated."""
    entry = lookup_offline_entry(word)
    return entry.model_dump(mode="json") if entry else None


AGENT_TOOLS = (score_spoken_answer, suggest_intended_words, lookup_offline_lexicon)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class HashingEmbedder(Embeddings):
    """Deterministic feature-hashing embedder (word + character n-grams).

    Lexical rather than semantic, but stable across processes and good enough
    to exercise the RAG path offline. Uses blake2b because Python's built-in
    ``hash`` is salted per process.
    """

    def __init__(self, dimensions: int = 384, ngram_range: tuple[int, int] = (3, 4)) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        self.dimensions = dimensions
        self.ngram_range = ngram_range

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _features(self, text: str) -> Iterator[str]:
        lo, hi = self.ngram_range
        for word in text.split():
            yield f"w:{word}"
            padded = f"<{word}>"
            for n in range(lo, hi + 1):
                for i in range(len(padded) - n + 1):
                    yield f"c:{padded[i:i + n]}"

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for feature in self._features(normalize_german(text, fold_umlauts=True)):
            value = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
            vector[value % self.dimensions] += 1.0 if value >> 63 else -1.0
        norm = float(np.linalg.norm(vector))
        return (vector / norm if norm else vector).tolist()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """You are the input pre-processor of a German learning app.
The input is either typed text or a speech-to-text transcript from an intermediate learner
(expect mispronunciations, missing umlauts, wrong capitalisation).

Decide:
- is_valid: true if the input is German or a recognisable mispronunciation / misspelling of German.
- normalized_input: the corrected input.
- root_form: the lemma. Infinitive for verbs (re-join separable verbs: "tritt ... ein" -> "eintreten"),
  nominative singular with capitalisation for nouns, base form for adjectives,
  canonical form for idioms ("verstehe nur Bahnhof" -> "nur Bahnhof verstehen").
- intent: "single_word", "phrase", "idiom", or "invalid".
- is_idiom: true for fixed idiomatic expressions whose meaning is not compositional.
- candidate_words: for voice input, up to 5 real German words the speaker plausibly meant, best first,
  including near-homophones (e.g. "eintretten" -> ["eintreten", "eintreffen", "betreten"]).
  Empty when the input is unambiguous.
- rejection_reason: a short reason when invalid, otherwise null."""

EXTRACTION_SYSTEM_PROMPT = """You are an expert German lexicographer writing for B1-C1 learners.
Produce a dictionary entry for the given lemma.

Rules:
1. contexts: one entry per distinct sense, most frequent first. Describe register, typical
   collocations, governed case/preposition, and auxiliary changes between senses.
   Each sense must be learnable on its own: never mix two senses in one context.
2. Each context has exactly two natural, grammatical example sentences with faithful translations.
3. used_conjugation must list the inflected headword exactly as it appears in the sentence, with
   separable parts in sentence order (e.g. "tritt ein", "ist eingetreten").
4. For verbs: fill conjugations (present, praeteritum, perfekt for all six persons, without pronouns)
   and set each example's tense; the used_conjugation must be a form of that tense.
   For non-verbs: conjugations is null and each example's tense is null.
5. tense_matrix (verbs only, else null): choose ONE short anchor sentence for the sense and render it in
   present, praeteritum and perfekt (exactly one row each). Keep subject, objects and adverbs identical;
   only the verb form may change. Use the auxiliary this sense requires (e.g. "hat ... eingetreten"
   for kicking in a door, "ist ... eingetreten" for entering).
6. homonyms: only distinct lexemes with the same spelling (different gender, plural, origin or
   stress, e.g. "die Bank" bench/bank, "umfahren" vs "umfahren"). Use an empty list when none exist."""

CRITIQUE_SYSTEM_PROMPT = """You are a strict examiner of German grammar reviewing a dictionary entry.
For every example sentence and every tense_matrix row check:
- the German sentence is grammatical and idiomatic (word order, case, agreement, separable prefixes);
- used_conjugation appears in the sentence and really is in the stated tense, with the correct auxiliary;
- the English translation is faithful;
- each sentence illustrates the sense of its own context and no other.
Report each problem with its zero-based context_index and example_index (use the row index for
tense_matrix rows and say so in the issue). Set is_valid to true only when there are no problems.
Do not report stylistic preferences."""

PRACTICE_SYSTEM_PROMPT = """You write targeted practice sentences for a German learner.
Write new sentences for exactly ONE sense of a word. Every sentence must be unambiguous: a reader
must not be able to interpret the word in any other sense. Do not reuse the reference sentences.
Rules: grammatical, natural B1-B2 German; used_conjugation is the inflected headword exactly as it
appears (separable parts in order); tense is set for verbs (null for non-verbs) and follows the
requested focus tenses; translations are faithful."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class LLMService:
    """Provider-agnostic facade for the agent tasks."""

    def __init__(
        self,
        settings: Settings,
        chat_model: BaseChatModel | None = None,
        *,
        anthropic_client: anthropic.Anthropic | None = None,
    ) -> None:
        """Injected clients win over settings (tests, or other LangChain providers such as Bedrock)."""
        self._settings = settings
        self._chat: BaseChatModel | None = chat_model
        self._claude: anthropic.Anthropic | None = anthropic_client
        self._structured_runnables: dict[type[BaseModel], Runnable] = {}
        if chat_model is not None or anthropic_client is not None:
            return

        provider = settings.active_provider
        if provider == "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
            self._claude = anthropic.Anthropic(
                api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_seconds, max_retries=2
            )
        elif provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
            self._chat = ChatOpenAI(
                model=settings.openai_model,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
                max_retries=2,
                api_key=settings.openai_api_key,
            )

    @property
    def is_online(self) -> bool:
        return self._chat is not None or self._claude is not None

    @property
    def mode(self) -> str:
        if self._claude is not None:
            return f"anthropic:{self._settings.anthropic_model}"
        if isinstance(self._chat, ChatOpenAI):
            return f"openai:{self._chat.model_name}"
        return type(self._chat).__name__ if self._chat is not None else "offline"

    def build_embedder(self) -> Embeddings:
        """OpenAI embeddings when a key is configured, else the deterministic hashing embedder.

        Anthropic has no embeddings endpoint, so a Claude-only setup keeps the
        hashing embedder (or Bedrock Titan embeddings once on AWS).
        """
        if self._settings.openai_api_key:
            return OpenAIEmbeddings(model=self._settings.openai_embedding_model, api_key=self._settings.openai_api_key)
        return HashingEmbedder()

    # -- agent tasks --------------------------------------------------------

    def analyze_intent(self, text: str, *, source: str = "text") -> IntentAnalysis:
        if not self.is_online:
            return self._offline_intent(text)
        return self._invoke_structured(
            IntentAnalysis, INTENT_SYSTEM_PROMPT, f"Input source: {source}\nInput: {text}"
        )

    def suggest_candidates(self, heard: str, *, extra_words: Iterable[str] = ()) -> list[WordCandidate]:
        """Levenshtein/phonetic ranking over the known vocabulary plus any model-proposed words.

        The model widens recall to the open vocabulary; the deterministic ranker
        decides the order, so scores stay comparable across both modes.
        """
        vocabulary = candidate_vocabulary()
        for word in extra_words:
            if word and word.strip():
                vocabulary.setdefault(word.strip().lower(), word.strip())
        return rank_candidates(heard, vocabulary)

    def extract_dictionary(
        self,
        root_form: str,
        *,
        is_idiom: bool = False,
        requested_tense: Tense | None = None,
        feedback: list[str] | None = None,
    ) -> DictionaryResponse:
        if not self.is_online:
            return self._offline_extract(root_form, requested_tense)

        parts = [f"Lemma: {root_form}", f"Is idiom: {is_idiom}"]
        if requested_tense:
            parts.append(f"Every example sentence must use the {requested_tense} tense (the tense_matrix still spans all three).")
        if feedback:
            parts.append(
                "Your previous draft was rejected by the grammar reviewer. Fix every issue:\n"
                + "\n".join(f"- {issue}" for issue in feedback)
            )
        return self._invoke_structured(DictionaryResponse, EXTRACTION_SYSTEM_PROMPT, "\n".join(parts))

    def critique_grammar(self, entry: DictionaryResponse, *, requested_tense: Tense | None = None) -> GrammarCritique:
        if not self.is_online:
            # Offline content is curated; the deterministic rule checks in the graph still run.
            return GrammarCritique(is_valid=True, issues=[])
        request = (
            f"Requested tense: {requested_tense or 'any'}\n"
            f"Entry:\n{json.dumps(entry.model_dump(mode='json'), ensure_ascii=False)}"
        )
        return self._invoke_structured(GrammarCritique, CRITIQUE_SYSTEM_PROMPT, request)

    def generate_practice_sentences(
        self,
        *,
        word: str,
        context: WordContext,
        conjugations: ConjugationSet | None,
        focus_tenses: Sequence[Tense] = (),
        count: int = 2,
    ) -> list[SentenceExample]:
        """New sentences that exercise *only* the given sense of ``word``."""
        if not self.is_online:
            return self._offline_practice_sentences(context, focus_tenses, count)

        references = [s.german for s in context.examples] + [v.german for v in context.tense_matrix or []]
        request = "\n".join(
            [
                f"Word: {word}",
                f"Target sense: {context.english_gloss} — {context.nuance_description}",
                f"Focus tenses: {', '.join(focus_tenses) if conjugations and focus_tenses else 'any'}",
                f"Number of sentences: {count}",
                "Reference sentences (do not repeat): " + " | ".join(references),
            ]
        )
        return self._invoke_structured(PracticeSentenceSet, PRACTICE_SYSTEM_PROMPT, request).sentences[:count]

    # -- provider plumbing --------------------------------------------------

    def _structured(self, schema: type[SchemaT]) -> Runnable:
        """LangChain path (OpenAI or an injected chat model)."""
        runnable = self._structured_runnables.get(schema)
        if runnable is None:
            assert self._chat is not None
            # OpenAI strict mode: the provider enforces the JSON schema at decode time.
            kwargs: dict[str, Any] = {"method": "json_schema", "strict": True} if isinstance(self._chat, ChatOpenAI) else {}
            runnable = self._chat.with_structured_output(schema, **kwargs)
            self._structured_runnables[schema] = runnable
        return runnable

    def _invoke_structured(self, schema: type[SchemaT], system_prompt: str, human_prompt: str) -> SchemaT:
        if self._claude is not None:
            return self._invoke_claude(schema, system_prompt, human_prompt)
        try:
            result = self._structured(schema).invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
        except Exception as exc:  # provider SDKs raise a wide, unstable set of exception types
            logger.exception("Structured generation failed for %s", schema.__name__)
            raise LLMServiceError(f"{schema.__name__} generation failed: {exc}") from exc

        if isinstance(result, schema):
            return result
        try:
            # Some providers return plain dicts instead of model instances.
            return schema.model_validate(result)
        except ValidationError as exc:
            raise LLMServiceError(f"{schema.__name__} output violated the schema: {exc}") from exc

    def _invoke_claude(self, schema: type[SchemaT], system_prompt: str, human_prompt: str) -> SchemaT:
        assert self._claude is not None
        model = self._settings.anthropic_model
        extra: dict[str, Any] = {}
        if not model.startswith("claude-haiku"):
            extra = {"betas": [CLAUDE_FALLBACK_BETA], "fallbacks": "default"}
        try:
            # No temperature/thinking params: current Claude models reject sampling
            # parameters, and thinking is adaptive by default; effort sets its depth.
            response = self._claude.beta.messages.parse(
                model=model,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": human_prompt}],
                output_format=schema,
                output_config={"effort": self._settings.anthropic_effort},
                **extra,
            )
        except anthropic.AuthenticationError as exc:
            raise LLMServiceError("Anthropic rejected the API key; check ANTHROPIC_API_KEY") from exc
        except anthropic.RateLimitError as exc:
            raise LLMServiceError(f"Anthropic rate limit reached: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            # 400s include "credit balance is too low" and invalid-model errors.
            logger.error("Anthropic %s for %s: %s (request %s)", exc.status_code, schema.__name__, exc.message, exc.request_id)
            raise LLMServiceError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMServiceError(f"Could not reach the Anthropic API: {exc}") from exc

        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise LLMServiceError(f"Claude declined the {schema.__name__} request (category: {category})")
        if response.stop_reason == "max_tokens":
            raise LLMServiceError(f"{schema.__name__} output was truncated at {CLAUDE_MAX_TOKENS} tokens")
        parsed = response.parsed_output
        if not isinstance(parsed, schema):
            raise LLMServiceError(f"Claude returned no valid {schema.__name__}")
        logger.info(
            "Claude %s for %s: %d in / %d out tokens",
            response.model, schema.__name__, response.usage.input_tokens, response.usage.output_tokens,
        )
        return parsed

    # -- offline fallbacks --------------------------------------------------

    @staticmethod
    def _offline_intent(text: str) -> IntentAnalysis:
        cleaned = " ".join(text.strip().strip(_EDGE_PUNCTUATION).split())
        if not cleaned or len(cleaned) > 120 or not _VALID_INPUT_RE.match(cleaned):
            return IntentAnalysis(
                is_valid=False, normalized_input=cleaned, root_form="", intent="invalid", is_idiom=False,
                candidate_words=[],
                rejection_reason="Input must be a German word or phrase (letters, spaces and hyphens only).",
            )

        lowered = cleaned.lower()
        if lowered in OFFLINE_IDIOMS:
            return IntentAnalysis(
                is_valid=True, normalized_input=cleaned, root_form=cleaned, intent="idiom", is_idiom=True,
                candidate_words=[], rejection_reason=None,
            )

        # Exact surface-form hit first, then the best fuzzy candidate.
        lemma = INFLECTION_INDEX.get(lowered)
        if lemma is None:
            candidates = rank_candidates(lowered, candidate_vocabulary(), limit=1)
            lemma = candidates[0].word if candidates else None
        return IntentAnalysis(
            is_valid=True,
            normalized_input=cleaned,
            root_form=lemma or cleaned,
            intent="single_word" if len(lowered.split()) == 1 or lemma else "phrase",
            is_idiom=False,
            candidate_words=[],
            rejection_reason=None,
        )

    @staticmethod
    def _offline_extract(root_form: str, requested_tense: Tense | None) -> DictionaryResponse:
        entry = lookup_offline_entry(root_form)
        if entry is None:
            available = ", ".join(sorted(e.word for e in OFFLINE_LEXICON.values()))
            raise LLMUnavailableError(
                f"'{root_form}' is not in the offline demo lexicon (available: {available}). "
                "Set OPENAI_API_KEY to enable live extraction."
            )
        if requested_tense is None:
            return entry

        contexts = []
        for context in entry.contexts:
            examples = [ex for ex in context.examples if ex.tense == requested_tense]
            if examples:
                contexts.append(context.model_copy(update={"examples": examples}))
        if not contexts:
            raise LLMUnavailableError(
                f"The offline entry for '{entry.word}' has no {requested_tense} examples. "
                "Set OPENAI_API_KEY to generate them."
            )
        return entry.model_copy(update={"contexts": contexts})

    @staticmethod
    def _offline_practice_sentences(
        context: WordContext, focus_tenses: Sequence[Tense], count: int
    ) -> list[SentenceExample]:
        """Offline stand-in: reuse this sense's own sentences, focus tenses first."""
        pool = [
            SentenceExample(german=v.german, english=v.english, used_conjugation=v.used_conjugation, tense=v.tense)
            for v in context.tense_matrix or []
        ] + list(context.examples)
        if focus_tenses:
            pool.sort(key=lambda s: s.tense not in focus_tenses)  # stable: keeps order within each group
        return pool[:count]
