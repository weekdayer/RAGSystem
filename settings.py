"""网页、HTTP 接口与语音组件共用的本地配置入口。"""

from dataclasses import dataclass, field
import os
from pathlib import Path

import dotenv


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass(frozen=True)
class RagSettings:
    api_key: str = field(repr=False)
    base_url: str
    chat_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v4"
    persist_directory: Path = ROOT / "data" / "chroma"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    relevance_threshold: float = 0.35
    faq_threshold: float = 0.70
    faq_min_margin: float = 0.08

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("请填写百炼的 OpenAI 兼容接口地址。")
        if not self.chat_model.strip() or not self.embedding_model.strip():
            raise ValueError("请填写对话模型和 Embedding 模型名称。")
        if self.chunk_size <= 0:
            raise ValueError("分块长度必须大于 0。")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("重叠长度必须大于等于 0，且小于分块长度。")
        if self.top_k <= 0:
            raise ValueError("检索数量必须大于 0。")
        if not 0 <= self.relevance_threshold <= 1:
            raise ValueError("相关度门槛必须在 0 到 1 之间。")
        if not 0 <= self.faq_threshold <= 1 or not 0 <= self.faq_min_margin <= 1:
            raise ValueError("FAQ 匹配门槛与领先差必须在 0 到 1 之间。")
        directory = Path(self.persist_directory)
        if not directory.is_absolute():
            directory = ROOT / directory
        object.__setattr__(self, "persist_directory", directory.resolve())


def load_environment() -> None:
    """已有进程环境变量优先；修改本地配置后重新启动服务。"""
    dotenv.load_dotenv(ROOT / ".env")


def load_rag_settings() -> RagSettings:
    """两入口共用环境配置；网页可在此结果上覆盖临时会话参数。"""
    load_environment()
    try:
        threshold = float(os.getenv("RAG_RELEVANCE_THRESHOLD", str(RagSettings.relevance_threshold)))
        top_k = int(os.getenv("RAG_TOP_K", str(RagSettings.top_k)))
        faq_threshold = float(os.getenv("FAQ_MATCH_THRESHOLD", str(RagSettings.faq_threshold)))
        faq_min_margin = float(os.getenv("FAQ_MIN_MARGIN", str(RagSettings.faq_min_margin)))
    except ValueError:
        raise ValueError("检索配置无效，请检查检索数量、相关度和 FAQ 门槛。") from None
    directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "").strip()
    return RagSettings(
        api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
        base_url=(os.getenv("DASHSCOPE_BASE_URL", "").strip() or DEFAULT_BASE_URL).rstrip("/"),
        chat_model=os.getenv("CHAT_MODEL", "").strip() or RagSettings.chat_model,
        embedding_model=os.getenv("EMBEDDING_MODEL", "").strip() or RagSettings.embedding_model,
        persist_directory=Path(directory) if directory else RagSettings.persist_directory,
        relevance_threshold=threshold,
        top_k=top_k,
        faq_threshold=faq_threshold,
        faq_min_margin=faq_min_margin,
    )
