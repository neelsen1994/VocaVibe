"""Voice-ready fuzzy matching: normalisation, Levenshtein, Kölner Phonetik, answer scoring.

Speech-to-text output from intermediate learners is noisy in two different
ways. Spelling noise ("eintretten") is caught by edit distance. Pronunciation
noise ("eintreeten", "aintreten") is caught by the Kölner Phonetik code,
which maps words that *sound* alike to the same digit string. Candidates are
ranked on a blend of both signals.

Pure Python, no dependencies. For a full-size vocabulary, swap
``levenshtein`` for ``rapidfuzz`` or push candidate generation into an
OpenSearch ``fuzzy`` query.
"""

import difflib
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

_UMLAUT_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
# Letter runs (umlauts included) with optional hyphenated compounds; drops digits and punctuation.
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")

# Ranking weights: spelling distance dominates; phonetics rescues mispronunciations.
LEXICAL_WEIGHT = 0.7
PHONETIC_WEIGHT = 0.3


def normalize_german(text: str, *, fold_umlauts: bool = False) -> str:
    """Lower-case, strip punctuation and collapse whitespace.

    ``fold_umlauts`` maps ä->ae, ö->oe, ü->ue, ß->ss. Speech-to-text engines and
    keyboards without German layouts emit both spellings, so comparisons
    should fold them.
    """
    lowered = unicodedata.normalize("NFC", text).lower()
    if fold_umlauts:
        lowered = lowered.translate(_UMLAUT_FOLD)
    return " ".join(_TOKEN_RE.findall(lowered))


def tokenize(text: str) -> list[str]:
    return normalize_german(text).split()


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert / delete / substitute), O(len(a) * len(b)) time, O(min) memory."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    """Edit distance scaled to 0..1 (1 = identical)."""
    longest = max(len(a), len(b))
    return 1.0 if longest == 0 else 1.0 - levenshtein(a, b) / longest


_COLOGNE_VOWELS = frozenset("aeiouyjäöü")
_C_INITIAL_HARD = frozenset("ahkloqrux")
_C_HARD = frozenset("ahkoqux")


def cologne_phonetic(text: str) -> str:
    """Kölner Phonetik code: German-specific phonetic hashing (like Soundex for English).

    "eintreten", "eintretten" and "eintreeten" all map to "062726".
    """
    word = normalize_german(text).replace(" ", "").replace("-", "")
    codes: list[str] = []
    for i, char in enumerate(word):
        prev = word[i - 1] if i > 0 else ""
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if char in _COLOGNE_VOWELS:
            code = "0"
        elif char == "b":
            code = "1"
        elif char == "p":
            code = "3" if nxt == "h" else "1"
        elif char in ("d", "t"):
            code = "8" if nxt in ("c", "s", "z", "ß") else "2"
        elif char in ("f", "v", "w"):
            code = "3"
        elif char in ("g", "k", "q"):
            code = "4"
        elif char == "c":
            if i == 0:
                code = "4" if nxt in _C_INITIAL_HARD else "8"
            else:
                code = "8" if prev in ("s", "z") else ("4" if nxt in _C_HARD else "8")
        elif char == "x":
            code = "8" if prev in ("c", "k", "q") else "48"
        elif char == "l":
            code = "5"
        elif char in ("m", "n"):
            code = "6"
        elif char == "r":
            code = "7"
        elif char in ("s", "z", "ß"):
            code = "8"
        else:  # 'h' and anything non-alphabetic carry no sound code
            code = ""
        codes.append(code)

    collapsed: list[str] = []
    for digit in "".join(codes):
        if not collapsed or collapsed[-1] != digit:
            collapsed.append(digit)
    result = "".join(collapsed)
    # Vowel codes are dropped everywhere except at the start of the word.
    return result[:1] + result[1:].replace("0", "")


# ---------------------------------------------------------------------------
# Voice-to-confirmation candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WordCandidate:
    word: str  # lemma to analyse if the learner confirms
    matched_form: str  # the surface form that matched, e.g. "tritt ein"
    score: float  # 0..1 blended similarity
    edit_distance: int
    phonetic_match: bool  # identical Kölner Phonetik code -> "sounds the same"
    correction: str  # human-readable fix, e.g. "drop 't'"


def describe_edits(heard: str, target: str) -> str:
    """Explain how to get from what was heard to the target, e.g. "'u' -> 'ü', drop 't'"."""
    heard_n, target_n = normalize_german(heard), normalize_german(target)
    edits: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, heard_n, target_n, autojunk=False).get_opcodes():
        if tag == "replace":
            edits.append(f"'{heard_n[i1:i2]}' -> '{target_n[j1:j2]}'")
        elif tag == "delete":
            edits.append(f"drop '{heard_n[i1:i2]}'")
        elif tag == "insert":
            edits.append(f"add '{target_n[j1:j2]}'")
    return ", ".join(edits) if edits else "exact match"


def rank_candidates(
    heard: str,
    vocabulary: Mapping[str, str],
    *,
    limit: int = 5,
    min_score: float = 0.6,
) -> list[WordCandidate]:
    """Rank the lemmas the learner most plausibly meant.

    ``vocabulary`` maps surface forms (any case) to lemmas, so inflected input
    such as "trat ein" still resolves to "eintreten". Only the best form per
    lemma is kept.
    """
    heard_folded = normalize_german(heard, fold_umlauts=True)
    if not heard_folded:
        return []
    heard_code = cologne_phonetic(heard)
    max_length_gap = max(3, len(heard_folded) // 2)

    best: dict[str, WordCandidate] = {}
    for form, lemma in vocabulary.items():
        form_folded = normalize_german(form, fold_umlauts=True)
        if not form_folded or abs(len(form_folded) - len(heard_folded)) > max_length_gap:
            continue  # cheap pre-filter before the O(n*m) distance
        distance = levenshtein(heard_folded, form_folded)
        lexical = 1.0 - distance / max(len(heard_folded), len(form_folded))
        form_code = cologne_phonetic(form)
        phonetic = 1.0 if form_code == heard_code else levenshtein_similarity(form_code, heard_code)
        score = round(LEXICAL_WEIGHT * lexical + PHONETIC_WEIGHT * phonetic, 4)
        if score < min_score:
            continue
        candidate = WordCandidate(
            word=lemma,
            matched_form=form,
            score=score,
            edit_distance=distance,
            phonetic_match=form_code == heard_code,
            correction=describe_edits(heard, form),
        )
        current = best.get(lemma)
        if current is None or (candidate.score, -candidate.edit_distance) > (current.score, -current.edit_distance):
            best[lemma] = candidate
    return sorted(best.values(), key=lambda c: (-c.score, c.edit_distance, c.word))[:limit]


# ---------------------------------------------------------------------------
# Answer scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FuzzyMatchResult:
    score: float  # 0..1 character-level similarity
    normalized_expected: str
    normalized_actual: str
    mismatches: tuple[str, ...]  # human-readable word-level differences


def fuzzy_compare(expected: str, actual: str) -> FuzzyMatchResult:
    """Compare a learner's answer (often a speech transcript) to the reference sentence."""
    norm_expected = normalize_german(expected, fold_umlauts=True)
    norm_actual = normalize_german(actual, fold_umlauts=True)
    if not norm_expected and not norm_actual:
        score = 1.0
    else:
        score = difflib.SequenceMatcher(None, norm_expected, norm_actual, autojunk=False).ratio()

    expected_tokens, actual_tokens = norm_expected.split(), norm_actual.split()
    mismatches: list[str] = []
    matcher = difflib.SequenceMatcher(None, expected_tokens, actual_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        wanted = " ".join(expected_tokens[i1:i2]) or "(nothing)"
        heard = " ".join(actual_tokens[j1:j2]) or "(nothing)"
        mismatches.append(f"expected '{wanted}' but got '{heard}'")
    return FuzzyMatchResult(round(score, 4), norm_expected, norm_actual, tuple(mismatches))


def contains_form(text: str, form: str, *, cutoff: float = 0.9) -> bool:
    """Whether every word of ``form`` (e.g. 'ist eingetreten') occurs in ``text``.

    Short words (auxiliaries, pronouns) must match exactly, since 'ist' vs 'isst'
    is a real error. Longer words tolerate minor recognition noise.
    """
    text_tokens = normalize_german(text, fold_umlauts=True).split()
    for part in normalize_german(form, fold_umlauts=True).split():
        if len(part) <= 4:
            if part not in text_tokens:
                return False
        elif not difflib.get_close_matches(part, text_tokens, n=1, cutoff=cutoff):
            return False
    return True
