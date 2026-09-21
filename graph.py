import base64
import os
import json
from typing import Literal, TypedDict

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field
from pathlib import Path

load_dotenv(find_dotenv(usecwd=True))
if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("blog-draft-agent 폴더의 .env 파일에 OPENAI_API_KEY 한 줄을 넣습니다.")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "blog-draft-agent"

# llm = init_chat_model("openai/gpt-5.6-luna", model_provider="litellm")
llm = init_chat_model("openai/gpt-4o-mini", model_provider="litellm")

emb = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory="chroma_db", embedding_function=emb)
vision_client = OpenAI()

PHOTO_ROOT = Path("meta")
SKIP_SUFFIXES = {".mp4", ".mov"}

DRY_RUN = True  # 임시: vision 호출 없이 코드 흐름만 확인


def retrieve_style(query: str, k: int = 3) -> list[str]:
    """과거 글 중 query와 비슷한 것 k개의 본문을 돌려준다."""
    hits = db.similarity_search(query, k=k)
    return [d.page_content for d in hits]


def describe_photo(path: Path) -> str:
    """사진 한 장을 블로그 본문에 쓸 수 있게 한두 문장으로 설명한다."""
    if DRY_RUN:
        return f"[사진 설명 생략(DRY_RUN): {path.name}]"

    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    mime = "image/gif" if path.suffix.lower() == ".gif" else "image/jpeg"
    res = vision_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "이 사진을 블로그 글에 쓸 수 있게 한두 문장으로 자연스럽게 설명해줘."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            ],
        }],
    )
    return res.choices[0].message.content.strip()


def describe_section(folder: Path) -> str:
    """폴더 안 사진들을 전부 설명해 이어 붙인다. 동영상·하위 폴더는 건너뛴다."""
    lines = []
    for photo in sorted(folder.iterdir()):
        if photo.is_dir() or photo.name.startswith(".") or photo.suffix.lower() in SKIP_SUFFIXES:
            continue
        lines.append(describe_photo(photo))
    return "\n".join(lines)


def describe_menu(folder: Path) -> str:
    """메뉴 폴더: 요리명 하위 폴더는 요리별로 묶어 설명하고, 가격 정보(meta.txt)를 같이 붙인다."""
    price_file = folder / "meta.txt"
    prices = price_file.read_text(encoding="utf-8") if price_file.exists() else "(가격 정보 없음)"

    dish_blocks = []
    for dish_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        photos = [p for p in sorted(dish_dir.iterdir())
                  if p.suffix.lower() not in SKIP_SUFFIXES and not p.name.startswith(".")]
        descs = "\n".join(describe_photo(p) for p in photos)
        dish_blocks.append(f"[{dish_dir.name}]\n{descs}")

    return f"=== 가격 정보 ===\n{prices}\n\n=== 요리별 사진 설명 ===\n" + "\n\n".join(dish_blocks)


class State(TypedDict):
    folders: dict           # 섹션 이름 -> 그 섹션의 사진 경로 목록
    meta: dict               # meta.txt를 읽어 만든 가게 정보 (가게 이름, 주소, post_type 등)
    post_type: str            # "내돈내산" 또는 "체험단"
    campaign_template: dict  # 체험단일 때 업체 요구 양식 (내돈내산이면 빈 dict)
    posting: str
    feedback: str
    grade: str
    tries: int


MAX_TRIES = 1


class Verdict(BaseModel):
    grade: Literal["합격", "반려"] = Field(description="글이 기준을 지켰는가")
    feedback: str = Field(description="반려라면 무엇을 고칠지 한 문장, 합격이면 빈 문자열")


def keyword_extract(state: State) -> dict:
    """meta 정보와 과거 글의 키워드 패턴을 참고해 이번 글의 키워드를 뽑는다."""
    meta = json.loads(Path("meta/meta.json").read_text(encoding="utf-8"))
    post_type = meta["post_type"]
    campaign_template = meta.get("campaign_template", {})

    query = meta["name"]
    posts = retrieve_style(query, k=3)
    past_posts_text = "\n---\n".join(posts)

    if post_type == "체험단":
        keyword_guide = f"'{campaign_template['필수 키워드']}' 중 1개를 키워드에 필수로 사용한다."
    else:
        keyword_guide = (
            "과거에 작성한 다음 글들의 '지역, 메뉴, 상황어 나열' 키워드 패턴을 따라, "
            f"이번 가게({meta['name']})에 어울리는 키워드를 만든다."
        )

    keyword = llm.invoke(
        f"{keyword_guide} 키워드는 각 10자 이내로 작성한다.\n\n"
        f"=== 과거 글 ===\n{past_posts_text}"
    ).content
    

    return {
            "meta": {**meta, 
            "keyword": keyword}, 
            "post_type":post_type,
            "campaign_template": campaign_template
            }


def title(state: State) -> dict:
    """키워드를 반영해 제목을 만든다."""
    meta = state["meta"]

    if meta["post_type"] == "체험단":
        title_guide = f"'{meta['keyword']}'를 반드시 포함해 제목을 만든다."
    else:
        title_guide = f"'{meta['keyword']}'를 자연스럽게 녹여 제목을 만든다."

    generated_title = llm.invoke(
        f"과거 작성한 글의 제목 패턴을 참고해 초안의 제목을 생성한다. {title_guide} "
        "제목에 특수문자를 넣지 않고, 20자를 넘지 않는다."
    ).content.strip()

    return {"meta": {**meta, "title": generated_title}}



def build_posting_instruction(state: State, extra_requirements: str = "") -> str:
    """write_posting과 write_posting_campaign이 함께 쓰는 지침 조립 함수."""
    meta = state["meta"]

    store_info = {k: v for k, v in meta.items() if k not in ("keyword", "title", "campaign_template")}

    sections_text = "\n\n".join([
        f"[대표사진]\n{describe_section(PHOTO_ROOT / '1_대표사진')}",
        f"[가게외관]\n{describe_section(PHOTO_ROOT / '2_가게외관')}",
        f"[매장 정보]\n{json.dumps(store_info, ensure_ascii=False)}",
        f"[매장분위기]\n{describe_section(PHOTO_ROOT / '3_매장분위기')}",
        f"[메뉴]\n{describe_menu(PHOTO_ROOT / '4_메뉴')}",
    ])

    past_posts = "\n---\n".join(retrieve_style(meta["name"], k=2))

    instruction = (
        f"아래 정보를 바탕으로 네이버 블로그 글 본문을 쓴다. 제목은 '{meta.get('title', '')}'이다.\n"
        "소제목([대표사진], [가게외관] 같은 이름 말고 자연스러운 소제목)과 문단 나누기를 적극 활용한다.\n"
        "첫 부분에는 이 가게를 방문하기 좋은 상황(맥락)을 자연스러운 문장으로 쓴다.\n"
        "사진을 설명한 문장 아래에는 핵심 정보(가격, 특징)를 한 번 더 텍스트로 짚어준다.\n"
        "키워드를 나열하지 말고, 직접 겪은 경험담 문장으로 자연스럽게 녹여 쓴다.\n"
        "마지막은 홍보성 문구나 클릭 유도 없이, 담백한 방문 팁으로 마무리한다.\n"
        f"{extra_requirements}\n\n"
        f"=== 참고할 과거 글(문체 참고용) ===\n{past_posts}\n\n"
        f"=== 이번 글 재료 ===\n{sections_text}"
    )

    if state["feedback"]:
        instruction += f"\n\n=== 지난 지적 (반드시 반영해서 다시 쓴다) ===\n{state['feedback']}"

    return instruction


def write_posting(state: State) -> dict:
    """내돈내산 글 본문을 쓴다. feedback이 있으면 반영해 다시 쓴다."""
    instruction = build_posting_instruction(state)
    result = llm.invoke(instruction).content
    return {"posting": result, "tries": state["tries"] + 1}


def write_posting_campaign(state: State) -> dict:
    """체험단 글 본문을 쓴다. campaign_template(업체 요구 양식)을 반영한다."""
    campaign = state["campaign_template"]
    extra_requirements = (
        "\n이 글은 체험단 원고다. 다음 업체 요구사항을 반드시 지킨다.\n"
        f"{json.dumps(campaign, ensure_ascii=False)}"
    )
    instruction = build_posting_instruction(state, extra_requirements=extra_requirements)
    result = llm.invoke(instruction).content
    return {"posting": result, "tries": state["tries"] + 1}


def check_tone(state: State) -> dict:
    """본문이 기준(문체, 필수 정보 포함 등)을 지켰는지 평가한다. 본문을 고치지 않는다."""
    grader = llm.with_structured_output(Verdict)

    verdict = grader.invoke(
        "다음 블로그 글이 아래 기준을 모두 지켰는지 판정한다. 하나라도 안 지켰으면 반려.\n"
        "- 소제목과 문단 구분이 있다.\n"
        "- 키워드를 나열하지 않고 직접 겪은 경험담 문장으로 녹여 썼다.\n"
        "- 마지막에 과도한 홍보 문구나 클릭 유도가 없다.\n"
        "- 사진을 설명한 문장 아래에 핵심 정보(가격 등)가 텍스트로 한 번 더 있다.\n\n"
        f"=== 글 ===\n{state['posting']}"
    )
    return {"grade": verdict.grade, "feedback": verdict.feedback}


def route_after_title(state: State) -> str:
    return "write_posting_campaign" if state["post_type"] == "체험단" else "write_posting"


def route_after_check(state: State) -> str:
    """합격이거나 시도 상한이면 끝내고, 아니면 아까 그 글을 쓴 노드로 되돌린다."""
    if state["grade"] == "합격" or state["tries"] >= MAX_TRIES:
        return END
    return "write_posting_campaign" if state["post_type"] == "체험단" else "write_posting"


g = StateGraph(State)
g.add_node("keyword_extract", keyword_extract)
g.add_node("title", title)
g.add_node("write_posting", write_posting)
g.add_node("write_posting_campaign", write_posting_campaign)
g.add_node("check_tone", check_tone)

g.add_edge(START, "keyword_extract")
g.add_edge("keyword_extract", "title")
g.add_conditional_edges("title", route_after_title, ["write_posting", "write_posting_campaign"])
g.add_edge("write_posting", "check_tone")
g.add_edge("write_posting_campaign", "check_tone")
g.add_conditional_edges("check_tone", route_after_check, ["write_posting", "write_posting_campaign", END])

graph = g.compile()

print("노드:", list(g.nodes))

result = graph.invoke({
    "folders": {},
    "meta": {},
    "post_type": "",
    "campaign_template": {},
    "posting": "",
    "feedback": "",
    "grade": "",
    "tries": 0,
})

Path("output.md").write_text(result["posting"], encoding="utf-8-sig")

