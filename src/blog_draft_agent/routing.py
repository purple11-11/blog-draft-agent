from langgraph.graph import END

from blog_draft_agent.state import MAX_TRIES, State


def route_after_title(state: State) -> str:
    return "write_posting_campaign" if state["post_type"] == "체험단" else "write_posting"


def route_after_check(state: State) -> str:
    """합격이거나 시도 상한이면 끝내고, 아니면 아까 그 글을 쓴 노드로 되돌린다."""
    if state["grade"] == "합격" or state["tries"] >= MAX_TRIES:
        return END
    return "write_posting_campaign" if state["post_type"] == "체험단" else "write_posting"
