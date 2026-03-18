from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..epub_agent import EPUBAgentState


def make_review_rules_node() -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    """Return a pass-through node that is the HITL interrupt point.

    Execution only reaches this node after ``approved_rules`` has been injected
    via ``graph.update_state()`` by the review API endpoint.  The node passes
    ``approved_rules`` through unchanged — the frontend receives rule strings
    directly from the backend so no re-validation is needed.
    """

    def review_rules(state: "EPUBAgentState") -> "EPUBAgentState":
        # Just propagate whatever the human submitted; apply_rules will use it.
        return {}

    return review_rules
