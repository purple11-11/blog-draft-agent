from typing import Literal, TypedDict

from pydantic import BaseModel, Field

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
