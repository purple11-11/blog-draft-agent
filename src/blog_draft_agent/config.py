import os

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
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
MAX_WORKERS = 4  # vision 호출을 동시에 몇 개까지 보낼지 (rate limit 고려)
