from blog_draft_agent.config import db


def retrieve_style(query: str, k: int = 3) -> list[str]:
    """과거 글 중 query와 비슷한 것 k개의 본문을 돌려준다."""
    hits = db.similarity_search(query, k=k)
    return [d.page_content for d in hits]