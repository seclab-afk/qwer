import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


# LLM 서버 설정
# LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://100.78.38.74:11434") # seclab2225
# LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://100.100.209.47:11434") # seclab2228
# LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://100.100.209.47:54321") # seclab2228 vllm
# LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://100.95.107.99:11434") # seclab2229
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://100.108.122.26:22229") # seclab2229 vllm

# LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-oss:20b")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-20b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# 호스트 경로 설정 (Docker 마운트용)
# HOST_BASE_PATH = os.getenv("HOST_BASE_PATH", "/data/shin/B/agent")
HOST_BASE_PATH = os.getenv("HOST_BASE_PATH", "/data/afk/agent")
