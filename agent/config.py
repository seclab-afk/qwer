import os
from dotenv import load_dotenv

load_dotenv()

LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "")  

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-20b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

HOST_BASE_PATH = os.getenv("HOST_BASE_PATH", "/path/to/afk/agent")
