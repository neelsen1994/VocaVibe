"""State channels shared by every node in the dictionary agent graph.

None of the channels has a reducer, so each node *overwrites* the keys it
returns. That is what the critique loop needs: ``critique_errors`` always
holds the verdict on the latest draft, not every draft so far.
"""

from typing import Any, NotRequired, TypedDict

from app.schemas.dictionary import IntentLabel, Tense


class AgentState(TypedDict):
    # --- core channels ---------------------------------------------------
    user_input: str
    analyzed_intent: IntentLabel | str  # "" until the intent node has run
    raw_dictionary_data: dict[str, Any]  # latest extraction draft
    validated_json: dict[str, Any]  # set only when the draft passes critique
    critique_errors: list[str]  # issues found in the latest draft
    iteration_count: int  # number of extraction attempts so far

    # --- request options and intent details --------------------------------
    input_source: NotRequired[str]  # "text" | "voice"
    confirmed: NotRequired[bool]  # learner already picked this word from a candidate list
    requested_tense: NotRequired[Tense | None]
    root_form: NotRequired[str]
    is_idiom: NotRequired[bool]
    rejection_reason: NotRequired[str | None]

    # --- voice-to-confirmation loop ------------------------------------------
    candidates: NotRequired[list[dict[str, Any]]]  # serialised WordCandidate rows
    needs_confirmation: NotRequired[bool]


def initial_state(
    user_input: str,
    *,
    source: str = "text",
    confirmed: bool = False,
    requested_tense: Tense | None = None,
) -> AgentState:
    """Graph entry state with every core channel initialised."""
    return AgentState(
        user_input=user_input,
        analyzed_intent="",
        raw_dictionary_data={},
        validated_json={},
        critique_errors=[],
        iteration_count=0,
        input_source=source,
        confirmed=confirmed,
        requested_tense=requested_tense,
        candidates=[],
        needs_confirmation=False,
    )
