import json
from pathlib import Path

from blog_draft_agent.config import llm
from blog_draft_agent.prompts import CLICHES, CLOSING_SIGNATURE, build_posting_instruction
from blog_draft_agent.retrieval import retrieve_style
from blog_draft_agent.state import State, Verdict


def keyword_extract(state: State) -> dict:
    """meta 정보와 과거 글의 키워드 패턴을 참고해 이번 글의 키워드를 뽑는다."""
    meta = json.loads(Path("meta/meta.json").read_text(encoding="utf-8"))
    post_type = meta["post_type"]
    campaign_template = meta.get("campaign_template", {})

    if post_type == "체험단":
        prompt = (
            f"'{campaign_template['필수 키워드']}' 중 1개를 키워드에 필수로 사용한다. "
            "키워드는 각 10자 이내로 작성한다."
        )
    else:
        posts = retrieve_style(f"{meta['name']} 맛집", k=3)
        past_posts_text = "\n---\n".join(posts)
        prompt = (
            "과거에 작성한 다음 글들의 '지역, 메뉴, 상황어 나열' 키워드 패턴(형식)만 참고해, "
            f"이번 가게({meta['name']})에 어울리는 키워드를 만든다. "
            "키워드에 들어갈 지역·메뉴는 이번 가게 이름에 있는 정보만 쓰고, "
            "참고 글에 나온 다른 가게의 이름이나 장소는 가져오지 않는다. "
            "키워드는 각 10자 이내로 작성한다.\n\n"
            f"=== 과거 글(형식만 참고) ===\n{past_posts_text}"
        )

    keyword = llm.invoke(prompt).content

    return {
        "meta": {**meta, "keyword": keyword},
        "post_type": post_type,
        "campaign_template": campaign_template,
    }


def title(state: State) -> dict:
    """키워드를 반영해 제목을 만든다."""
    meta = state["meta"]

    if meta["post_type"] == "체험단":
        title_guide = f"'{meta['keyword']}'를 반드시 포함해 제목을 만든다."
    else:
        title_guide = f"'{meta['keyword']}'를 자연스럽게 녹여 제목을 만든다."

    generated_title = llm.invoke(
        f"과거 작성한 글의 제목 패턴을 참고해 초안의 제목을 생성한다. "
        f"가게 이름('{meta['name']}')을 반드시 포함한다. {title_guide} "
        "제목에 특수문자를 넣지 않고, 20자를 넘지 않는다."
    ).content.strip()

    return {"meta": {**meta, "title": generated_title}}



def write_posting(state: State) -> dict:
    """내돈내산 글 본문을 쓴다. feedback이 있으면 반영해 다시 쓴다."""
    instruction = build_posting_instruction(state)
    title_line = f"# {state['meta'].get('title', '')}\n\n"
    result = title_line + llm.invoke(instruction).content + CLOSING_SIGNATURE
    return {"posting": result, "tries": state["tries"] + 1}


def write_posting_campaign(state: State) -> dict:
    """체험단 글 본문을 쓴다. campaign_template(업체 요구 양식)을 반영한다."""
    meta = state["meta"]
    posting_length = state["campaign_template"].get("포스팅 분량", "")
    extra_requirements = (
        "\n이 글은 체험단 원고다. "
        f"필수 키워드는 이미 '{meta['keyword']}'로 정해졌으니 이 키워드 하나만 자연스럽게 한 번 넣는다 "
        "(업체가 준 필수 키워드 목록 전체를 억지로 다 넣지 않는다). "
        f"분량 요구사항: {posting_length}"
    )
    instruction = build_posting_instruction(state, extra_requirements=extra_requirements)
    title_line = f"# {meta.get('title', '')}\n\n"
    result = title_line + llm.invoke(instruction).content + CLOSING_SIGNATURE
    return {"posting": result, "tries": state["tries"] + 1}


def check_tone(state: State) -> dict:
    """본문이 기준(문체, 필수 정보 포함 등)을 지켰는지 평가한다. 본문을 고치지 않는다."""
    grader = llm.with_structured_output(Verdict)

    verdict = grader.invoke(
        "다음 블로그 글이 아래 기준을 모두 지켰는지 판정한다. 하나라도 안 지켰으면 반려.\n"
        "- '매장 분위기', 'Menu', '마무리' 헤딩과 '✔' 라벨이 그대로 있고, "
        "템플릿에 없는 문장이나 섹션을 추가하지 않았다.\n"
        "- '비추천' 목록이나 '지도' 관련 내용이 없다.\n"
        "- 메뉴 항목이 실제 재료(요리명·가격)와 일치한다.\n"
        f"- 상투적이고 감상적인 미사여구({CLICHES})가 없다.\n"
        "- 긴 문단이 아니라 사진(재료 항목)당 1~2줄 짧은 캡션 형식으로 끊어 썼다.\n"
        "- '전체적으로 좋았습니다'처럼 실질적 정보가 없는 채우기용 문장이 없다.\n"
        "- 보이는 사물을 전부 나열하지 않고, 인상적인 것 한두 개만 골라 반응을 붙였다.\n\n"
        f"=== 글 ===\n{state['posting']}"
    )
    return {"grade": verdict.grade, "feedback": verdict.feedback}

