"""Curated offline lexicon and the candidate vocabulary for voice confirmation.

The full entries back offline mode and the tests. The lemma list gives the
voice-confirmation loop realistic near-misses ("eintretten" -> eintreten /
eintreffen / betreten). In production the vocabulary would come from a
frequency list kept in S3 or a DynamoDB lexicon table.
"""

from typing import Any

from app.schemas.dictionary import TENSES, DictionaryResponse

_OFFLINE_ENTRIES: list[dict[str, Any]] = [
    {
        "word": "eintreten",
        "part_of_speech": "verb",
        "homonyms": [],
        "conjugations": {
            "infinitive": "eintreten",
            "auxiliary": "haben_or_sein",
            "partizip_ii": "eingetreten",
            "present": {
                "ich": "trete ein", "du": "trittst ein", "er_sie_es": "tritt ein",
                "wir": "treten ein", "ihr": "tretet ein", "sie_Sie": "treten ein",
            },
            "praeteritum": {
                "ich": "trat ein", "du": "tratst ein", "er_sie_es": "trat ein",
                "wir": "traten ein", "ihr": "tratet ein", "sie_Sie": "traten ein",
            },
            "perfekt": {
                "ich": "bin eingetreten", "du": "bist eingetreten", "er_sie_es": "ist eingetreten",
                "wir": "sind eingetreten", "ihr": "seid eingetreten", "sie_Sie": "sind eingetreten",
            },
        },
        "contexts": [
            {
                "nuance_description": (
                    "Physically entering a room or building (intransitive, Perfekt with 'sein'). "
                    "Common as a polite invitation; more formal than 'reinkommen'."
                ),
                "english_gloss": "to enter, to step in",
                "examples": [
                    {"german": "Bitte treten Sie ein!", "english": "Please come in!",
                     "used_conjugation": "treten ein", "tense": "present"},
                    {"german": "Sie trat leise in das Zimmer ein.", "english": "She quietly entered the room.",
                     "used_conjugation": "trat ein", "tense": "praeteritum"},
                ],
                "tense_matrix": [
                    {"tense": "present", "german": "Er tritt leise in das Zimmer ein.",
                     "english": "He quietly enters the room.", "used_conjugation": "tritt ein"},
                    {"tense": "praeteritum", "german": "Er trat leise in das Zimmer ein.",
                     "english": "He quietly entered the room.", "used_conjugation": "trat ein"},
                    {"tense": "perfekt", "german": "Er ist leise in das Zimmer eingetreten.",
                     "english": "He has quietly entered the room.", "used_conjugation": "ist eingetreten"},
                ],
            },
            {
                "nuance_description": (
                    "Joining an organisation, club or party: 'in etwas (Akk.) eintreten' (Perfekt with 'sein'). "
                    "The opposite is 'austreten'."
                ),
                "english_gloss": "to join",
                "examples": [
                    {"german": "Mein Bruder ist letztes Jahr in den Verein eingetreten.",
                     "english": "My brother joined the club last year.",
                     "used_conjugation": "ist eingetreten", "tense": "perfekt"},
                    {"german": "Wann trittst du in die Gewerkschaft ein?", "english": "When are you joining the union?",
                     "used_conjugation": "trittst ein", "tense": "present"},
                ],
                "tense_matrix": [
                    {"tense": "present", "german": "Sie tritt in die Partei ein.",
                     "english": "She joins the party.", "used_conjugation": "tritt ein"},
                    {"tense": "praeteritum", "german": "Sie trat in die Partei ein.",
                     "english": "She joined the party.", "used_conjugation": "trat ein"},
                    {"tense": "perfekt", "german": "Sie ist in die Partei eingetreten.",
                     "english": "She has joined the party.", "used_conjugation": "ist eingetreten"},
                ],
            },
            {
                "nuance_description": (
                    "Coming to pass, said of events, conditions or cases ('der Fall tritt ein'). "
                    "Formal register, frequent in news and legal texts."
                ),
                "english_gloss": "to occur, to set in",
                "examples": [
                    {"german": "Der befürchtete Fall ist nicht eingetreten.",
                     "english": "The feared scenario did not occur.",
                     "used_conjugation": "ist eingetreten", "tense": "perfekt"},
                    {"german": "Plötzlich trat eine lange Stille ein.", "english": "Suddenly a long silence set in.",
                     "used_conjugation": "trat ein", "tense": "praeteritum"},
                ],
                "tense_matrix": [
                    {"tense": "present", "german": "Der Ernstfall tritt ein.",
                     "english": "The emergency occurs.", "used_conjugation": "tritt ein"},
                    {"tense": "praeteritum", "german": "Der Ernstfall trat ein.",
                     "english": "The emergency occurred.", "used_conjugation": "trat ein"},
                    {"tense": "perfekt", "german": "Der Ernstfall ist eingetreten.",
                     "english": "The emergency has occurred.", "used_conjugation": "ist eingetreten"},
                ],
            },
            {
                "nuance_description": "Advocating for someone or something: 'für jemanden/etwas eintreten'.",
                "english_gloss": "to stand up for, to advocate",
                "examples": [
                    {"german": "Sie tritt seit Jahren für den Umweltschutz ein.",
                     "english": "She has been standing up for environmental protection for years.",
                     "used_conjugation": "tritt ein", "tense": "present"},
                    {"german": "Wir sind immer für faire Löhne eingetreten.",
                     "english": "We have always advocated for fair wages.",
                     "used_conjugation": "sind eingetreten", "tense": "perfekt"},
                ],
                "tense_matrix": [
                    {"tense": "present", "german": "Er tritt für seine Kollegen ein.",
                     "english": "He stands up for his colleagues.", "used_conjugation": "tritt ein"},
                    {"tense": "praeteritum", "german": "Er trat für seine Kollegen ein.",
                     "english": "He stood up for his colleagues.", "used_conjugation": "trat ein"},
                    {"tense": "perfekt", "german": "Er ist für seine Kollegen eingetreten.",
                     "english": "He has stood up for his colleagues.", "used_conjugation": "ist eingetreten"},
                ],
            },
            {
                "nuance_description": (
                    "Transitive: kicking in a door or window. In this sense the Perfekt switches to 'haben' "
                    "('Er hat die Tür eingetreten')."
                ),
                "english_gloss": "to kick in, to break down",
                "examples": [
                    {"german": "Er tritt die Tür mit voller Wucht ein.", "english": "He kicks the door in with full force.",
                     "used_conjugation": "tritt ein", "tense": "present"},
                    {"german": "Die Feuerwehr trat die Tür ein.", "english": "The fire brigade kicked the door in.",
                     "used_conjugation": "trat ein", "tense": "praeteritum"},
                ],
                "tense_matrix": [
                    {"tense": "present", "german": "Er tritt die Tür ein.",
                     "english": "He kicks the door in.", "used_conjugation": "tritt ein"},
                    {"tense": "praeteritum", "german": "Er trat die Tür ein.",
                     "english": "He kicked the door in.", "used_conjugation": "trat ein"},
                    {"tense": "perfekt", "german": "Er hat die Tür eingetreten.",
                     "english": "He has kicked the door in.", "used_conjugation": "hat eingetreten"},
                ],
            },
        ],
    },
    {
        "word": "Bank",
        "part_of_speech": "noun",
        "homonyms": [
            {"word": "Bank", "article": "die", "part_of_speech": "noun",
             "short_definition": "bench, seat (plural: die Bänke)"},
            {"word": "Bank", "article": "die", "part_of_speech": "noun",
             "short_definition": "bank, financial institution (plural: die Banken)"},
        ],
        "conjugations": None,
        "contexts": [
            {
                "nuance_description": "Seating: a bench in a park, station or church. Plural 'die Bänke' with umlaut.",
                "english_gloss": "bench",
                "examples": [
                    {"german": "Wir sitzen auf einer Bank im Park.", "english": "We are sitting on a bench in the park.",
                     "used_conjugation": "Bank", "tense": None},
                    {"german": "Im Bahnhof gibt es zu wenige Bänke.", "english": "There are too few benches in the station.",
                     "used_conjugation": "Bänke", "tense": None},
                ],
                "tense_matrix": None,
            },
            {
                "nuance_description": "Finance: a bank as an institution or branch. Plural 'die Banken' without umlaut.",
                "english_gloss": "bank (financial)",
                "examples": [
                    {"german": "Ich muss heute noch zur Bank gehen.", "english": "I still have to go to the bank today.",
                     "used_conjugation": "Bank", "tense": None},
                    {"german": "Viele Banken haben ihre Gebühren erhöht.", "english": "Many banks have raised their fees.",
                     "used_conjugation": "Banken", "tense": None},
                ],
                "tense_matrix": None,
            },
        ],
    },
]

# Validated at import time: a malformed demo entry fails fast, not at request time.
OFFLINE_LEXICON: dict[str, DictionaryResponse] = {
    entry["word"].lower(): DictionaryResponse.model_validate(entry) for entry in _OFFLINE_ENTRIES
}

OFFLINE_IDIOMS = frozenset(
    {"ich verstehe nur bahnhof", "tomaten auf den augen haben", "da steppt der bär", "das ist nicht mein bier"}
)

# Frequent lemmas, weighted towards minimal pairs learners mix up when speaking.
COMMON_LEMMAS: tuple[str, ...] = (
    # verbs: the -treten / -treffen / -tragen family and other prefix minimal pairs
    "treten", "eintreten", "betreten", "auftreten", "austreten", "vertreten", "zurücktreten",
    "treffen", "eintreffen", "betreffen", "übertreffen", "tragen", "eintragen", "vertragen",
    "fahren", "umfahren", "erfahren", "einsteigen", "aussteigen", "umsteigen",
    "gehen", "kommen", "bekommen", "ankommen", "stehen", "verstehen", "bestehen", "entstehen",
    "sehen", "aussehen", "nehmen", "annehmen", "übernehmen", "geben", "ergeben", "vergeben",
    "sprechen", "versprechen", "besprechen", "sagen", "fragen", "lernen", "arbeiten", "machen",
    "denken", "bedenken", "schlafen", "laufen", "kaufen", "verkaufen", "warten", "erwarten",
    "halten", "erhalten", "ziehen", "erziehen", "beziehen", "bringen", "verbringen",
    "setzen", "übersetzen", "legen", "überlegen", "stellen", "vorstellen", "bestellen",
    "suchen", "besuchen", "versuchen", "finden", "bleiben", "schreiben", "beschreiben",
    "lesen", "essen", "trinken", "wohnen", "gewöhnen", "leben", "erleben", "lieben",
    "helfen", "glauben", "erlauben", "hoffen",
    # nouns, including common homonyms
    "Bank", "Schloss", "Kiefer", "Leiter", "Band", "Tor", "See", "Gericht", "Strauß",
    "Tür", "Zimmer", "Stuhl", "Tisch", "Haus", "Straße", "Zeit", "Weg", "Kind", "Arbeit",
    "Stadt", "Land", "Wasser", "Brot", "Geld", "Freund", "Schule", "Sprache", "Wort", "Satz",
    # adjectives
    "groß", "klein", "gut", "schlecht", "schön", "schnell", "langsam", "wichtig", "richtig", "fertig",
)


def _build_inflection_index() -> dict[str, str]:
    """Map every known surface form (lower-case) to its lemma."""
    index: dict[str, str] = {}
    for entry in OFFLINE_LEXICON.values():
        index[entry.word.lower()] = entry.word
        if entry.conjugations:
            index[entry.conjugations.partizip_ii.lower()] = entry.word
            for tense in TENSES:
                for form in entry.conjugations.forms_for(tense).all_forms():
                    index.setdefault(form.lower(), entry.word)
        for context in entry.contexts:
            for sentence in [*context.examples, *(context.tense_matrix or [])]:
                index.setdefault(sentence.used_conjugation.lower(), entry.word)
    return index


INFLECTION_INDEX: dict[str, str] = _build_inflection_index()


def candidate_vocabulary() -> dict[str, str]:
    """Surface form -> lemma for every form the confirmation loop can propose."""
    vocabulary = {lemma.lower(): lemma for lemma in COMMON_LEMMAS}
    vocabulary.update(INFLECTION_INDEX)
    return vocabulary


def lookup_offline_entry(word: str) -> DictionaryResponse | None:
    entry = OFFLINE_LEXICON.get(word.strip().lower())
    return entry.model_copy(deep=True) if entry else None
