import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "AJgW6HFlwGktOyPL6tjP_RalzeTactS64-x5wFb2yVc")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LLM_MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-small"
CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 4
CANDIDATES = 20   # how many results each retriever fetches before fusion
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TRACING_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
# Reasoning effort: low | medium | high | routed
DEFAULT_EFFORT = os.getenv("REASONING_EFFORT", "medium")
ROUTE_LOW = os.getenv("ROUTE_LOW", "low")
ROUTE_HIGH = os.getenv("ROUTE_HIGH", "high")
