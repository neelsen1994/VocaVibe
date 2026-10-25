"""LangGraph workflow: intent analysis -> dictionary extraction <-> grammar critique.

    START -> intent_analysis --(invalid)----------------> END
                   |  --(voice, not an exact word)-------> END  (needs_confirmation + candidates)
                   v
        dictionary_extraction <----------------+
                   |                           | errors and retries left
                   v                           |
            grammar_critique ------------------+
                   |
                   v  (clean, or retry budget spent)
                  END

Voice-to-confirmation: a transcript that is not an exact known word never gets
silently auto-corrected. The graph stops and returns ranked candidates, and the
client re-submits the learner's pick with ``confirmed=True``. That keeps the
flow stateless; no checkpointer or session affinity is needed.

Critique runs in two layers. The deterministic checks in
``app.agents.validation`` go first; the LLM critic is only called on drafts
that pass them, which keeps cost down.
"""

import logging
from dataclasses import asdict
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from app.agents.state import AgentState
from app.agents.validation import find_rule_violations
from app.schemas.dictionary import DictionaryResponse
from app.services.fuzzy_service import normalize_german
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

NODE_INTENT = "intent_analysis"
NODE_EXTRACT = "dictionary_extraction"
NODE_CRITIQUE = "grammar_critique"


def build_dictionary_graph(llm: LLMService, *, max_retries: int = 2) -> CompiledStateGraph:
    """Compile the agent graph. ``max_retries`` caps the critique -> extraction loop-backs."""
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    def intent_node(state: AgentState) -> dict[str, Any]:
        raw = state["user_input"]
        analysis = llm.analyze_intent(raw, source=state.get("input_source", "text"))
        if not analysis.is_valid:
            return {"analyzed_intent": "invalid", "rejection_reason": analysis.rejection_reason, "candidates": []}

        heard = normalize_german(raw, fold_umlauts=True)
        # Model proposals widen recall, but echoing the raw input back must not
        # count as "the word exists"; only the vocabulary can confirm an exact hit.
        corrections = [
            w for w in (analysis.normalized_input, *analysis.candidate_words)
            if normalize_german(w, fold_umlauts=True) != heard
        ]
        lemma = [analysis.root_form] if normalize_german(analysis.root_form, fold_umlauts=True) != heard else []
        candidates = llm.suggest_candidates(raw, extra_words=[*corrections, *lemma])
        exact = any(c.edit_distance == 0 for c in candidates)
        # Online, the model also recognises words outside our vocabulary: if it neither
        # corrected the input nor proposed alternatives, the learner said a real word.
        # (A differing root_form is lemmatisation, e.g. "geht los" -> "losgehen", not a correction.)
        model_confident = llm.is_online and not corrections
        needs_confirmation = (
            state.get("input_source") == "voice"
            and not state.get("confirmed", False)
            and not exact
            and not model_confident
            and bool(candidates)
        )
        logger.info(
            "Intent for %r: %s (root=%r, idiom=%s, confirm=%s)",
            raw, analysis.intent, analysis.root_form, analysis.is_idiom, needs_confirmation,
        )
        return {
            "analyzed_intent": analysis.intent,
            "root_form": analysis.root_form or analysis.normalized_input,
            "is_idiom": analysis.is_idiom,
            "rejection_reason": None,
            "candidates": [asdict(c) for c in candidates],
            "needs_confirmation": needs_confirmation,
        }

    def extraction_node(state: AgentState) -> dict[str, Any]:
        attempt = state["iteration_count"] + 1
        entry = llm.extract_dictionary(
            state["root_form"],
            is_idiom=state.get("is_idiom", False),
            requested_tense=state.get("requested_tense"),
            feedback=state["critique_errors"] or None,
        )
        logger.info("Extraction attempt %d for %r produced %d contexts", attempt, state["root_form"], len(entry.contexts))
        return {"raw_dictionary_data": entry.model_dump(mode="json"), "iteration_count": attempt, "validated_json": {}}

    def critique_node(state: AgentState) -> dict[str, Any]:
        requested_tense = state.get("requested_tense")
        try:
            entry = DictionaryResponse.model_validate(state["raw_dictionary_data"])
        except ValidationError as exc:
            errors = [f"schema: {err['loc']}: {err['msg']}" for err in exc.errors()]
            return {"critique_errors": errors, "validated_json": {}}

        errors = find_rule_violations(entry, requested_tense)
        if not errors:
            critique = llm.critique_grammar(entry, requested_tense=requested_tense)
            errors = [
                f"contexts[{i.context_index}].examples[{i.example_index}]: {i.issue} (suggested: {i.suggested_fix})"
                for i in critique.issues
            ]
            if not critique.is_valid and not errors:
                errors = ["grammar critic rejected the entry without itemised issues"]

        if errors:
            logger.warning("Critique rejected attempt %d: %s", state["iteration_count"], errors)
            return {"critique_errors": errors, "validated_json": {}}
        return {"critique_errors": [], "validated_json": entry.model_dump(mode="json")}

    def route_after_intent(state: AgentState) -> str:
        if state["analyzed_intent"] == "invalid" or state.get("needs_confirmation"):
            return END
        return NODE_EXTRACT

    def route_after_critique(state: AgentState) -> str:
        if not state["critique_errors"]:
            return END
        # iteration_count counts extractions: 1 initial pass + up to max_retries repairs.
        return NODE_EXTRACT if state["iteration_count"] <= max_retries else END

    graph = StateGraph(AgentState)
    graph.add_node(NODE_INTENT, intent_node)
    graph.add_node(NODE_EXTRACT, extraction_node)
    graph.add_node(NODE_CRITIQUE, critique_node)

    graph.add_edge(START, NODE_INTENT)
    graph.add_conditional_edges(NODE_INTENT, route_after_intent, {NODE_EXTRACT: NODE_EXTRACT, END: END})
    graph.add_edge(NODE_EXTRACT, NODE_CRITIQUE)
    graph.add_conditional_edges(NODE_CRITIQUE, route_after_critique, {NODE_EXTRACT: NODE_EXTRACT, END: END})
    return graph.compile()
