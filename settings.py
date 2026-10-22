"""网页、HTTP 接口与语音组件共用的本地配置入口。"""

import os
from pathlib import Path

import dotenv

from rag import RagSettings


ROOT = Path(__file__).resolve().parent


def load_environment() -> None:
    """已有进程环境变量优先；修改本地配置后重新启动服务。"""
    dotenv.load_dotenv(ROOT / ".env")


def faq_matching_options() -> dict[str, float]:
    """网页与接口共用 FAQ 匹配配置，不影响文档检索门槛。"""
    try:
        return {
            "faq_threshold": float(os.getenv("FAQ_MATCH_THRESHOLD", "0.70")),
            "faq_min_margin": float(os.getenv("FAQ_MIN_MARGIN", "0.08")),
        }
    except ValueError:
        raise ValueError("FAQ 匹配配置无效，请检查门槛和领先差。") from None


def load_rag_settings() -> RagSettings:
    load_environment()
    try:
        threshold = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.35"))
        top_k = int(os.getenv("RAG_TOP_K", "4"))
    except ValueError:
        raise ValueError("检索配置无效，请检查相关度门槛和检索数量。") from None
    return RagSettings(
        api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip().rstrip("/"),
        chat_model=os.getenv("CHAT_MODEL", "qwen-plus").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4").strip(),
        persist_directory=Path(
            os.getenv("CHROMA_PERSIST_DIRECTORY", str(ROOT / "data" / "chroma"))
        ),
        relevance_threshold=threshold,
        top_k=top_k,
        **faq_matching_options(),
    )
