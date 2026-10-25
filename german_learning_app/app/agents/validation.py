"""Deterministic linguistic checks, shared by the critique node and practice generation.

These run before (and instead of) any LLM critic call, so they must be cheap
and must never reject correct German. They only check internal consistency
between a sentence, its declared verb form and the entry's conjugation table.
"""

from typing import Protocol

from app.schemas.dictionary import TENSES, ConjugationSet, DictionaryResponse, Tense, WordContext
from app.services.fuzzy_service import tokenize

HABEN_FORMS = frozenset({"habe", "hast", "hat", "haben", "habt"})
SEIN_FORMS = frozenset({"bin", "bist", "ist", "sind", "seid"})
# Minimum word overlap (Jaccard, verb excluded) between rows of a tense matrix.
# Below this the rows no longer show "the same sentence in another tense".
MATRIX_CONTEXT_OVERLAP = 0.6


class ConjugatedSentence(Protocol):
    german: str
    used_conjugation: str
    tense: Tense | None


def _matches_tense(used_tokens: list[str], conjugations: ConjugationSet, tense: Tense) -> bool:
    if tense == "perfekt":
        # Validate auxiliary + participle structurally so verbs whose auxiliary depends
        # on the sense (eintreten: "ist eingetreten" vs "hat die Tür eingetreten") pass.
        participle = set(tokenize(conjugations.partizip_ii))
        auxiliaries = [t for t in used_tokens if t not in participle]
        allowed = {"haben": HABEN_FORMS, "sein": SEIN_FORMS}.get(conjugations.auxiliary, HABEN_FORMS | SEIN_FORMS)
        return participle <= set(used_tokens) and len(auxiliaries) == 1 and auxiliaries[0] in allowed
    table = {t for form in conjugations.forms_for(tense).all_forms() for t in tokenize(form)}
    return set(used_tokens) <= table


def sentence_violations(sentence: ConjugatedSentence, conjugations: ConjugationSet | None, where: str) -> list[str]:
    """Check that the declared form occurs in the sentence and agrees with its tense."""
    used_tokens = tokenize(sentence.used_conjugation)
    if not used_tokens:
        return [f"{where}: used_conjugation is empty"]

    errors: list[str] = []
    sentence_tokens = set(tokenize(sentence.german))
    missing = [t for t in used_tokens if t not in sentence_tokens]
    if missing:
        errors.append(
            f"{where}: used_conjugation '{sentence.used_conjugation}' does not appear in "
            f"'{sentence.german}' (missing: {', '.join(missing)})"
        )
    if conjugations is None:
        return errors
    if sentence.tense is None:
        errors.append(f"{where}: verb sentences must declare a tense")
    elif not _matches_tense(used_tokens, conjugations, sentence.tense):
        errors.append(f"{where}: '{sentence.used_conjugation}' is not a valid {sentence.tense} form")
    return errors


def matrix_violations(context: WordContext, conjugations: ConjugationSet | None, where: str) -> list[str]:
    """A verb context needs one anchor sentence rendered in each tense, with only the verb changing."""
    matrix = context.tense_matrix
    if conjugations is None:
        return [f"{where}.tense_matrix: must be null for non-verbs"] if matrix else []
    if not matrix:
        return [f"{where}.tense_matrix: required for verbs"]

    errors: list[str] = []
    tenses = [variant.tense for variant in matrix]
    for tense in TENSES:
        if tenses.count(tense) != 1:
            errors.append(f"{where}.tense_matrix: needs exactly one {tense} row (found {tenses.count(tense)})")
    for vi, variant in enumerate(matrix):
        errors.extend(sentence_violations(variant, conjugations, f"{where}.tense_matrix[{vi}]"))

    frames = [set(tokenize(v.german)) - set(tokenize(v.used_conjugation)) for v in matrix]
    for vi, frame in enumerate(frames[1:], start=1):
        union = frame | frames[0]
        if union and len(frame & frames[0]) / len(union) < MATRIX_CONTEXT_OVERLAP:
            errors.append(
                f"{where}.tense_matrix[{vi}]: changes the sentence context; only the verb form may differ "
                f"from '{matrix[0].german}'"
            )
    return errors


def find_rule_violations(entry: DictionaryResponse, requested_tense: Tense | None = None) -> list[str]:
    """All deterministic checks for a full dictionary entry."""
    errors: list[str] = []
    if not entry.contexts:
        errors.append("contexts: at least one context is required")
    if entry.part_of_speech == "verb" and entry.conjugations is None:
        errors.append("conjugations: required when part_of_speech is 'verb'")

    for ci, context in enumerate(entry.contexts):
        where = f"contexts[{ci}]"
        if not context.examples:
            errors.append(f"{where}: at least one example is required")
        for ei, example in enumerate(context.examples):
            errors.extend(sentence_violations(example, entry.conjugations, f"{where}.examples[{ei}]"))
            # The matrix always spans every tense, so requested_tense applies to examples only.
            if requested_tense and entry.conjugations and example.tense != requested_tense:
                errors.append(f"{where}.examples[{ei}]: uses {example.tense} but {requested_tense} was requested")
        errors.extend(matrix_violations(context, entry.conjugations, where))
    return errors
