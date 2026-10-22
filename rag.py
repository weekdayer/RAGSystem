"""最小 RAG：读取文件、分块、持久化检索，再交给百炼模型回答。"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import logging
import math
from pathlib import Path
from collections.abc import Iterator, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from prompts import build_assistant_system_prompt, build_query_rewrite_system_prompt
from settings import RagSettings


logger = logging.getLogger(__name__)


def load_document(name: str, content: bytes) -> list[Document]:
    """解析 TXT、Markdown 或文字型 PDF；不写入上传的原文件。"""
    source = Path(name.replace("\\", "/")).name
    extension = Path(source).suffix.lower()
    if extension not in {".txt", ".md", ".pdf"}:
        raise ValueError("暂时只支持 .txt、.md 和 .pdf 文件。")
    if not content:
        raise ValueError("文件是空的，请上传有文字内容的文件。")

    if extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("PDF 已加密。")
            documents = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": source, "page": page_number},
                        )
                    )
        except Exception as exc:
            raise ValueError("无法读取 PDF，请确认文件未损坏且未加密。") from exc
        if not documents:
            raise ValueError("PDF 中没有可提取的文字；扫描件需要先进行 OCR。")
        return documents

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content.decode("gb18030")
        except UnicodeDecodeError as exc:
            raise ValueError("无法识别文本编码，请将文件保存为 UTF-8 后重试。") from exc
    if "\x00" in text:
        raise ValueError("文件含有二进制内容，请上传 UTF-8 或 GB18030 编码的文本。")
    if not text.strip():
        raise ValueError("文件中没有可用的文字内容。")
    return [Document(page_content=text.strip(), metadata={"source": source})]


def _prepare_history(history: list[dict] | None) -> list[tuple[str, str]]:
    """只保留最近 12 条对话，并限制发送给模型的历史长度。"""
    messages = []
    remaining = 18000
    for message in reversed((history or [])[-12:]):
        if not isinstance(message, dict):
            continue
        role, content = message.get("role"), message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()[: min(4000, remaining)]
        if content:
            messages.append((role, content))
            remaining -= len(content)
        if remaining == 0:
            break
    return list(reversed(messages))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float | None:
    """使用真实向量算余弦；零向量、维数不符或非有限值不可作引用依据。"""
    if not len(left) or len(left) != len(right):
        return None
    left_norm, right_norm = math.hypot(*left), math.hypot(*right)
    if (
        not left_norm
        or not right_norm
        or not math.isfinite(left_norm)
        or not math.isfinite(right_norm)
    ):
        return None
    score = math.fsum(
        (float(a) / left_norm) * (float(b) / right_norm)
        for a, b in zip(left, right)
    )
    return max(-1.0, min(1.0, score))


class RAGService:
    def __init__(
        self,
        settings: RagSettings,
        *,
        embeddings: Embeddings | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        if (embeddings is None or llm is None) and not settings.api_key.strip():
            raise ValueError("请先填写阿里云百炼 API Key。")
        self.settings = settings
        self.embeddings = embeddings if embeddings is not None else OpenAIEmbeddings(
            model=settings.embedding_model.strip(),
            api_key=settings.api_key.strip(),
            base_url=settings.base_url.strip().rstrip("/"),
            chunk_size=10,
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
            timeout=60,
            max_retries=2,
        )
        self.llm = llm if llm is not None else ChatOpenAI(
            model=settings.chat_model.strip(),
            api_key=settings.api_key.strip(),
            base_url=settings.base_url.strip().rstrip("/"),
            temperature=0,
            extra_body={"enable_thinking": False},
            use_responses_api=False,
            timeout=60,
            max_retries=2,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True,
        )

        # 模型或分块配置不同，使用独立集合，避免混用已有向量。
        index_config = {
            "version": 1,
            "base_url": settings.base_url.strip().rstrip("/"),
            "embedding_model": settings.embedding_model.strip(),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }
        digest = sha256(json.dumps(index_config, sort_keys=True).encode()).hexdigest()
        self.collection_name = f"rag_{digest[:24]}"
        self.client = chromadb.PersistentClient(
            path=str(settings.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        from faq import FAQService

        faq_config = {
            "version": 1,
            "base_url": settings.base_url.strip().rstrip("/"),
            "embedding_model": settings.embedding_model.strip(),
        }
        faq_digest = sha256(json.dumps(faq_config, sort_keys=True).encode()).hexdigest()
        self.faq = FAQService(
            settings.persist_directory / "faq.sqlite3",
            client=self.client,
            embeddings=self.embeddings,
            collection_name=f"faq_{faq_digest[:24]}",
            threshold=settings.faq_threshold,
            min_margin=settings.faq_min_margin,
        )
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=build_assistant_system_prompt(settings.chat_model)
                ),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )
        self.chat_prompt = chat_prompt
        self.chat_chain = chat_prompt | self.llm | StrOutputParser()
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=build_assistant_system_prompt(
                        settings.chat_model, with_context=True
                    )
                ),
                MessagesPlaceholder("history"),
                ("human", "本轮本地参考资料：\n{context}\n\n当前问题：{question}"),
            ]
        )
        self.answer_prompt = prompt
        self.answer_chain = prompt | self.llm | StrOutputParser()
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=build_query_rewrite_system_prompt()),
                MessagesPlaceholder("history"),
                ("human", "当前问题：{question}"),
            ]
        )
        self.query_rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()

    def count(self) -> int:
        """当前配置对应的知识库分块数。"""
        return self.client.get_collection(self.collection_name).count()

    def add_file(self, name: str, content: bytes) -> int:
        """加入缺少的分块；相同文件名和内容重复上传时返回 0。"""
        documents = load_document(name, content)
        chunks = self.splitter.split_documents(documents)
        source = documents[0].metadata["source"]
        file_id = sha256(source.encode("utf-8") + b"\x00" + content).hexdigest()
        ids = [f"{file_id}_{index}" for index in range(len(chunks))]
        existing = set(self.vector_store.get(ids=ids, include=[])["ids"])
        missing = []
        for index, (chunk, chunk_id) in enumerate(zip(chunks, ids)):
            if chunk_id not in existing:
                chunk.metadata.update({"file_id": file_id, "chunk_index": index})
                missing.append((chunk_id, chunk))

        # 小批提交；中途失败后重试只补齐缺块，不覆盖已成功写入的数据。
        for start in range(0, len(missing), 10):
            batch = missing[start : start + 10]
            self.vector_store.add_documents(
                documents=[chunk for _, chunk in batch],
                ids=[chunk_id for chunk_id, _ in batch],
            )
        return len(missing)

    def _retrieve(
        self, question: str, history: list[tuple[str, str]]
    ) -> list[Document]:
        count = self.count()
        if not count:
            return []
        search_question = question
        if history:
            try:
                rewritten = self.query_rewrite_chain.invoke(
                    {"question": question, "history": history}
                ).strip()
                if rewritten and len(rewritten) <= 4000:
                    search_question = rewritten
            except Exception as exc:
                logger.warning("检索问题改写失败：%s", type(exc).__name__)

        query_vector = self.embeddings.embed_query(search_question)
        if _cosine_similarity(query_vector, query_vector) is None:
            return []
        # 旧集合使用 L2 距离；扩大候选后按真实余弦重排，无需迁移现有资料。
        fetch_k = min(count, max(20, self.settings.top_k * 4))
        results = self.client.get_collection(self.collection_name).query(
            query_embeddings=[query_vector],
            n_results=fetch_k,
            include=["documents", "metadatas", "embeddings"],
        )
        vectors = results["embeddings"]
        if vectors is None:
            return []
        documents = []
        for text, metadata, vector in zip(
            results["documents"][0], results["metadatas"][0], vectors[0]
        ):
            score = _cosine_similarity(query_vector, vector)
            if text and score is not None and score >= self.settings.relevance_threshold:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={**(metadata or {}), "relevance_score": score},
                    )
                )
        documents.sort(key=lambda doc: doc.metadata["relevance_score"], reverse=True)
        return documents[: self.settings.top_k]

    def _prepare_answer(
        self, question: str, history: list[dict] | None = None
    ) -> tuple[Runnable | None, dict, list[Document]]:
        """普通与流式回答共用同一次检索及提示准备。"""
        question = question.strip()
        if not question:
            raise ValueError("请先输入问题。")
        if len(question) > 4000:
            raise ValueError("问题过长，请缩短到 4000 个字符以内。")
        try:
            faq_match = self.faq.match(question)
            if faq_match is not None:
                return None, {"answer": faq_match.page_content}, [faq_match]
        except Exception as exc:
            logger.warning("FAQ 检索失败，继续原问答流程：%s", type(exc).__name__)
        messages = _prepare_history(history)
        try:
            documents = self._retrieve(question, messages)
        except Exception as exc:
            # 不记录异常正文，避免远端错误中带出密钥或用户资料。
            logger.warning("本地资料检索失败，转为普通对话：%s", type(exc).__name__)
            documents = []
        if not documents:
            return self.chat_chain, {"question": question, "history": messages}, []
        context_parts = []
        for number, document in enumerate(documents, start=1):
            source = str(document.metadata.get("source", "未知文件"))
            page = document.metadata.get("page")
            location = f"，第 {page} 页" if page is not None else ""
            context_parts.append(
                f"[{number}] 来源：{source}{location}\n{document.page_content}"
            )
        return (
            self.answer_chain,
            {
                "question": question,
                "context": "\n\n".join(context_parts),
                "history": messages,
            },
            documents,
        )

    def ask(
        self, question: str, history: list[dict] | None = None
    ) -> tuple[str, list[Document]]:
        """结合有限历史聊天，仅将达到相关度门槛的本地资料交给模型。"""
        chain, inputs, documents = self._prepare_answer(question, history)
        if chain is None:
            return inputs["answer"], documents
        answer = chain.invoke(inputs)
        if not answer.strip():
            raise ValueError("模型没有返回有效回答，请重试。")
        return answer, documents

    def ask_stream(
        self, question: str, history: list[dict] | None = None
    ) -> Iterator[dict]:
        """先发送资料，再转发模型实际产生的增量文本，最后发送完整回答。"""
        chain, inputs, documents = self._prepare_answer(question, history)
        yield {
            "type": "sources",
            "sources": [
                {"content": document.page_content, "metadata": dict(document.metadata)}
                for document in documents
            ],
            "mode": "faq" if chain is None else "knowledge" if documents else "model",
        }
        if chain is None:
            # 标准答案已有全文，直接通过相同事件协议返回，不额外调用模型。
            yield {"type": "token", "content": inputs["answer"]}
            yield {"type": "done", "answer": inputs["answer"]}
            return
        chunks = []
        prompt = self.answer_prompt if documents else self.chat_prompt
        prompt_value = prompt.invoke(inputs)
        parser = StrOutputParser()
        # 直接持有模型迭代器，避免 RunnableSequence 关闭时为追踪而耗尽后续文本。
        stream = self.llm.stream(prompt_value)
        try:
            for chunk in stream:
                text = parser.invoke(chunk)
                if text:
                    chunks.append(text)
                    yield {"type": "token", "content": text}
        finally:
            # 消费方断开或主动关闭迭代时，也释放底层模型流。
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    logger.warning("关闭模型流失败：%s", type(exc).__name__)
        answer = "".join(chunks)
        if not answer.strip():
            raise ValueError("模型没有返回有效回答，请重试。")
        yield {"type": "done", "answer": answer}
