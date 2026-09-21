import base64
import os
import json
import re
from typing import Literal, TypedDict

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
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

llm = init_chat_model("openai/gpt-5.6-luna", model_provider="litellm")
# llm = init_chat_model("openai/gpt-4o-mini", model_provider="litellm")

emb = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory="chroma_db", embedding_function=emb)
vision_client = OpenAI()

PHOTO_ROOT = Path("meta")
SKIP_SUFFIXES = {".mp4", ".mov"}

DRY_RUN = False  # 임시: vision 호출 없이 코드 흐름만 확인


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
    meta: dict               # meta.json을 읽어 만든 가게 정보 (가게 이름, post_type, keyword, title 등)
    post_type: str            # "내돈내산" 또는 "체험단"
    campaign_template: dict  # 체험단일 때 업체 요구 양식 (내돈내산이면 빈 dict)
    posting: str
    feedback: str
    grade: str
    tries: int


MAX_TRIES = 3


class Verdict(BaseModel):
    grade: Literal["합격", "반려"] = Field(description="글이 기준을 지켰는가")
    feedback: str = Field(description="반려라면 무엇을 고칠지 한 문장, 합격이면 빈 문자열")


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



CLOSING_SIGNATURE = "\n\n끝까지 읽어주셔서 감사합니다!\n다들 밥 꼭 챙겨드세요~🍚"

TEMPLATE_SKELETON = """<방문 요일·시간, 웨이팅 여부를 담백한 한 문장으로만 쓴다. 미사여구 없이 사실만 적는다.>

📍 주소
📞 전화
⏰ 영업시간
🚗 주차
✅ 방문 시기 :

## 매장 분위기

## Menu
저희는

✔ <메뉴명> (₩ <가격>)
✔ <메뉴명> (₩ <가격>) <수량이 있으면 x숫자>

주문했습니다.

## 마무리
✔ 이런 사람에게 추천
<메뉴·분위기 재료를 바탕으로 2~3개, '-'로 시작하는 목록>"""


CLICHES = "'발걸음을 멈추게 하다', '안성맞춤이다', '~을 놓치지 마세요' 등"

STYLE_RULES = (
    f"- 상투적이고 감상적인 미사여구({CLICHES})는 쓰지 않는다.\n"
    "- 곁들이는 반찬·조리 방식·맛·식감처럼 재료에 없는 세부 사실은 지어내지 않는다. "
    "메뉴 종류만 보고 조리법이나 맛을 통념으로 단정하지 않는다.\n"
    "- 같은 문장 어미나 단어·표현 조합을 여러 문단에서 반복하지 않는다.\n"
    "- '참고할 과거 글'은 실제 문체 예시다. 존댓말/반말 섞임, 이모티콘·'ㅋㅋ' 사용 여부, "
    "문장 길이와 끊어 쓰는 방식을 그대로 따라 쓰되, 그 글의 가게 이름·장소는 가져오지 않는다.\n"
)


def build_posting_instruction(state: State, extra_requirements: str = "") -> str:
    """write_posting과 write_posting_campaign이 함께 쓰는 지침 조립 함수."""
    meta = state["meta"]

    sections_text = "\n\n".join([
        f"[매장분위기]\n{describe_section(PHOTO_ROOT / '3_매장분위기')}",
        f"[메뉴]\n{describe_menu(PHOTO_ROOT / '4_메뉴')}",
    ])

    past_posts = "\n---\n".join(retrieve_style(meta["keyword"], k=2))

    instruction = (
        f"아래 정보로 네이버 블로그 글 본문을 쓴다. 제목은 '{meta.get('title', '')}'이다.\n"
        "아래 템플릿의 헤딩과 '✔'·'📍' 같은 라벨은 그대로 쓰고 <> 안만 채운다. "
        "템플릿에 없는 문장·섹션, '지도' 관련 내용, '비추천' 목록은 추가하지 않는다.\n\n"
        f"=== 템플릿 ===\n{TEMPLATE_SKELETON}\n\n"
        "메뉴 항목은 '이번 글 재료'의 요리명·가격 정보만 그대로 쓴다.\n"
        f"{STYLE_RULES}"
        f"{extra_requirements}\n\n"
        f"=== 참고할 과거 글(말투만 참고, 내용은 무시) ===\n{past_posts}\n\n"
        f"=== 이번 글 재료 ===\n{sections_text}"
    )

    if state["feedback"]:
        instruction += f"\n\n=== 지난 지적 (반드시 반영해서 다시 쓴다) ===\n{state['feedback']}"

    return instruction


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
        f"- 상투적이고 감상적인 미사여구({CLICHES})가 없다.\n\n"
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
    "meta": {},
    "post_type": "",
    "campaign_template": {},
    "posting": "",
    "feedback": "",
    "grade": "",
    "tries": 0,
})

def next_output_path(root: Path = Path(".")) -> Path:
    """output.md, output2.md, ... 중 가장 큰 번호 다음 파일 경로를 돌려준다."""
    numbers = [0]
    for p in root.glob("output*.md"):
        m = re.fullmatch(r"output(\d*)\.md", p.name)
        if m:
            numbers.append(int(m.group(1)) if m.group(1) else 1)
    next_n = max(numbers) + 1
    return Path(f"output{next_n}.md")


out_path = next_output_path()
out_path.write_text(result["posting"], encoding="utf-8-sig")
print("저장 위치:", out_path)

