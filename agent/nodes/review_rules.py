from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from ..epub_agent import EPUBAgentState


def make_review_rules_node() -> Callable[["EPUBAgentState"], "EPUBAgentState"]:
    """Return a pass-through validation node that is the HITL interrupt point.

    Execution only reaches this node after ``approved_rules`` has been injected
    via ``graph.update_state()`` by the review API endpoint.  The node validates
    that every approved rule is a known cleanup rule (dropping unknown entries).
    """

    def review_rules(state: "EPUBAgentState") -> "EPUBAgentState":
        approved: List[str] = list(state.get("approved_rules") or [])
        known: set = set(state.get("cleanup_rules") or [])
        valid = [r for r in approved if r in known]
        return {"approved_rules": valid}

    return review_rules
