import os
from typing import Literal, TypedDict

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(usecwd=True))
if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("blog-draft-agent 폴더의 .env 파일에 OPENAI_API_KEY 한 줄을 넣습니다.")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "blog-draft-agent"

llm = init_chat_model("openai/gpt-5.6-luna", model_provider="litellm")

# index_corpus.ipynb에서 만든 벡터 저장소를 그대로 불러와 재사용한다 (다시 임베딩하지 않는다).
emb = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory="chroma-db", embedding_function=emb)


def retrieve_style(query: str, k: int = 3) -> list[str]:
    """과거 글 중 query와 비슷한 것 k개의 본문을 돌려준다."""
    hits = db.similarity_search(query, k=k)
    return [d.page_content for d in hits]


class State(TypedDict):
    folders: dict           # 섹션 이름 -> 그 섹션의 사진 경로 목록
    meta: dict               # meta.txt를 읽어 만든 가게 정보 (가게 이름, 주소, post_type 등)
    post_type: str            # "내돈내산" 또는 "체험단"
    campaign_template: str    # 체험단일 때 업체 요구 양식 (내돈내산이면 빈 문자열)
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
    # TODO 1: retrieve_style(...)로 meta["가게 이름"]이나 메뉴로 검색해 과거 글 몇 개를 가져온다.
    # TODO 2: 가져온 과거 글 + state["meta"]를 llm에 넣어서 "이 가게 이름/메뉴/상황을 담은 키워드 목록"을 뽑는다.
    #         (본인 과거 글의 "지역+메뉴+상황어 나열" 패턴을 따라 쓰라고 지침을 준다.)
    keywords = ...  # TODO: llm.invoke([...]) 결과에서 뽑은 키워드 리스트
    return {"meta": {**state["meta"], "keywords": keywords}}


def title(state: State) -> dict:
    """키워드를 반영해 제목을 만든다."""
    # TODO: state["meta"]["keywords"]를 넣고, 본인 과거 제목 패턴을 따라 쓰라는 지침으로 llm을 호출한다.
    generated_title = ...  # TODO
    return {"meta": {**state["meta"], "title": generated_title}}


def write_posting(state: State) -> dict:
    """내돈내산 글 본문을 쓴다. feedback이 있으면 반영해 다시 쓴다."""
    # TODO 1: retrieve_style(...)로 문체 참고용 과거 글을 가져온다.
    # TODO 2: state["folders"](섹션별 사진 설명 필요하면 여기서 vision 호출), state["meta"] 정보를 모아
    #         "1_대표사진 -> 2_가게외관 -> 지도매장정보(meta) -> 매장분위기 -> 메뉴 -> 마무리" 순서로 글을 쓰라는 지침을 만든다.
    # TODO 3: state["feedback"]이 비어있지 않으면, 그 지적을 반영해 다시 쓰라는 지침을 추가한다. (6강 ex06 generator 패턴 참고)
    result = ...  # TODO: llm.invoke([...])
    return {"posting": result, "tries": state["tries"] + 1}


def write_posting_campaign(state: State) -> dict:
    """체험단 글 본문을 쓴다. campaign_template(업체 요구 양식)을 반영한다."""
    # TODO: write_posting과 거의 같지만, state["campaign_template"] 내용을 지침에 포함시켜야 한다.
    result = ...  # TODO
    return {"posting": result, "tries": state["tries"] + 1}


def check_tone(state: State) -> dict:
    """본문이 기준(문체, 필수 정보 포함 등)을 지켰는지 평가한다. 본문을 고치지 않는다."""
    # TODO: llm.with_structured_output(Verdict)를 만들고, 뭘 기준으로 합격/반려를 가릴지 지침을 정해 판정을 받는다.
    #       (6강 ex06 evaluator 패턴 참고 — grader.invoke([...]))
    verdict = ...  # TODO
    return {"grade": verdict.grade, "feedback": verdict.feedback}


def route_after_title(state: State) -> str:
    return "write_posting_campaign" if state["post_type"] == "체험단" else "write_posting"


def route_after_check(state: State) -> str:
    """합격이거나 시도 상한이면 끝내고, 아니면 아까 그 글을 쓴 노드로 되돌린다."""
    # TODO: state["grade"], state["tries"], MAX_TRIES를 보고
    #       END로 보낼지, write_posting/write_posting_campaign 중 어디로 되돌릴지 정한다.
    #       (되돌릴 노드는 route_after_title과 같은 기준으로 정하면 된다.)
    ...  # TODO: return END 또는 "write_posting" 또는 "write_posting_campaign"


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
