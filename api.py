"""Local FastAPI interface for the shared RAG knowledge base.

Run with: python -m uvicorn api:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints
from starlette.concurrency import run_in_threadpool


logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_FAQ_UPLOAD_BYTES = 1024 * 1024
Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]
FAQAnswer = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)
]
FAQScore = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
FAQCandidateScore = Annotated[float, Field(strict=True, ge=-1, le=1, allow_inf_nan=False)]


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: Question


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    history: list[HistoryMessage] = Field(default_factory=list, max_length=100)


class Source(BaseModel):
    content: str
    metadata: dict[str, JsonValue]


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: Literal["faq", "knowledge", "model"]


class FAQCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    answer: FAQAnswer
    aliases: list[Question] = Field(default_factory=list, max_length=20)


class FAQItem(BaseModel):
    id: str
    question: str
    answer: str
    aliases: list[str]
    updated_at: str


class FAQListResponse(BaseModel):
    items: list[FAQItem]
    count: int


class FAQImportResponse(BaseModel):
    added: int = Field(ge=0)
    skipped: int = Field(ge=0)
    count: int = Field(ge=0)


class FAQSettingsResponse(BaseModel):
    threshold: FAQScore
    min_margin: FAQScore


class FAQSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: FAQScore
    min_margin: FAQScore | None = None


class FAQMatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question


class FAQCandidate(BaseModel):
    id: str
    question: str
    semantic_score: FAQCandidateScore
    bm25_score: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
    hybrid_score: FAQCandidateScore


class FAQMatchedItem(BaseModel):
    id: str
    question: str
    answer: str
    match_method: Literal["exact", "semantic"]


class FAQMatchResponse(BaseModel):
    question: str
    matched: bool
    reason: Literal["exact", "hybrid", "empty", "below_threshold", "ambiguous", "no_candidates"]
    threshold: FAQScore
    min_margin: FAQScore
    match: FAQMatchedItem | None
    candidates: list[FAQCandidate]


class KnowledgeStatus(BaseModel):
    count: int


class DocumentResponse(BaseModel):
    filename: str
    added_chunks: int
    count: int


class TranscriptionResponse(BaseModel):
    text: str


app = FastAPI(title="本地智能助手 API", version="0.1.0")


@app.exception_handler(RequestValidationError)
async def invalid_request(_request, _error):
    # FastAPI's default details can echo arbitrary input, including credentials.
    return JSONResponse(status_code=400, content={"detail": "请求参数无效，请检查输入字段。"})


def _safe_error(error: Exception, operation: str) -> tuple[int, str]:
    status = getattr(error, "status_code", None)
    logger.warning(
        "API操作失败：阶段=%s，异常类型=%s，上游状态码=%s",
        operation,
        type(error).__name__,
        status if isinstance(status, int) else None,
    )
    if isinstance(error, (APITimeoutError, APIConnectionError)):
        return 503, "模型服务暂时不可用，请稍后重试。"
    if isinstance(error, APIStatusError):
        if error.status_code == 429 or error.status_code >= 500:
            return 503, "模型服务暂时不可用，请稍后重试。"
        return 502, "上游模型服务拒绝了请求，请检查服务端配置。"
    if isinstance(error, ValueError):
        return 400, "请求内容无效，请检查输入、文件格式或文档内容。"
    return 500, "服务处理失败，请稍后重试。"


def _raise_service_error(error: Exception, operation: str) -> None:
    status, message = _safe_error(error, operation)
    raise HTTPException(status_code=status, detail=message) from None


@lru_cache(maxsize=1)
def get_rag_service():
    """Build a single shared local service only when a RAG endpoint needs it."""
    try:
        from rag import RAGService
        from settings import load_rag_settings

        return RAGService(load_rag_settings())
    except Exception as error:
        logger.warning("问答服务初始化失败：异常类型=%s", type(error).__name__)
        raise HTTPException(status_code=503, detail="问答服务尚未配置或初始化失败。") from None


@lru_cache(maxsize=1)
def get_transcriber():
    try:
        from speech import create_transcriber

        return create_transcriber()
    except Exception as error:
        logger.warning("语音服务初始化失败：异常类型=%s", type(error).__name__)
        raise HTTPException(status_code=503, detail="语音服务尚未配置或初始化失败。") from None


def _filename(name: str | None, default: str) -> str:
    return Path((name or default).replace("\\", "/")).name or default


def _read_upload(upload: UploadFile, *, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    size_message = (
        "FAQ 文件超过 1 MiB，请缩小文件后重试。"
        if max_bytes == MAX_FAQ_UPLOAD_BYTES
        else "上传文件超过 20 MiB，请缩小文件后重试。"
    )
    try:
        if upload.size is not None and upload.size > max_bytes:
            raise HTTPException(status_code=413, detail=size_message)
        content = upload.file.read(max_bytes + 1)
    except HTTPException:
        raise
    except Exception as error:
        _raise_service_error(error, "读取上传文件")
    finally:
        upload.file.close()
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=size_message)
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    return content


def _source(content: Any, metadata: Any) -> Source:
    if not isinstance(content, str) or not isinstance(metadata, dict):
        raise TypeError("Invalid source data")
    if any(not isinstance(key, str) for key in metadata):
        raise TypeError("Invalid metadata keys")
    # Normalize to JSON values and reject NaN/Infinity before response rendering.
    try:
        normalized = json.loads(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError):
        raise TypeError("Source metadata must contain JSON values") from None
    return Source(content=content, metadata=normalized)


def _sources(documents) -> list[Source]:
    return [_source(document.page_content, document.metadata) for document in documents]


def _answer_mode(sources: list[Source]) -> Literal["faq", "knowledge", "model"]:
    if any(source.metadata.get("kind") == "faq" for source in sources):
        return "faq"
    return "knowledge" if sources else "model"


def _history(payload: ChatRequest) -> list[dict]:
    return [message.model_dump() for message in payload.history]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/knowledge/status", response_model=KnowledgeStatus)
def knowledge_status(service=Depends(get_rag_service)):
    try:
        return KnowledgeStatus(count=service.count())
    except Exception as error:
        _raise_service_error(error, "知识库状态")


@app.post("/documents", response_model=DocumentResponse)
def documents(file: UploadFile = File(...), service=Depends(get_rag_service)):
    name = _filename(file.filename, "document")
    if Path(name).suffix.lower() not in {".txt", ".md", ".pdf"}:
        file.file.close()
        raise HTTPException(status_code=400, detail="仅支持 TXT、Markdown 或文字型 PDF。")
    content = _read_upload(file)
    try:
        added = service.add_file(name, content)
        return DocumentResponse(filename=name, added_chunks=added, count=service.count())
    except Exception as error:
        _raise_service_error(error, "文档加入")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, service=Depends(get_rag_service)):
    try:
        answer, documents = service.ask(payload.question, history=_history(payload))
        sources = _sources(documents)
        return ChatResponse(answer=answer, sources=sources, mode=_answer_mode(sources))
    except Exception as error:
        _raise_service_error(error, "问答")


def _stream_event(event: dict) -> dict:
    kind = event.get("type")
    if kind == "sources":
        sources = [_source(item["content"], item["metadata"]) for item in event["sources"]]
        return {
            "type": "sources",
            "sources": [source.model_dump() for source in sources],
            "mode": _answer_mode(sources),
        }
    if kind == "token" and isinstance(event.get("content"), str):
        return {"type": "token", "content": event["content"]}
    if kind == "done" and isinstance(event.get("answer"), str):
        return {"type": "done", "answer": event["answer"]}
    raise TypeError("Invalid streaming event")


def _sse(event: dict) -> str:
    payload = json.dumps(event, ensure_ascii=False, allow_nan=False)
    return f"event: {event['type']}\ndata: {payload}\n\n"


def _close_iterator(iterator) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        try:
            close()
        except Exception as error:
            logger.warning("关闭回答流失败：异常类型=%s", type(error).__name__)


class ClosingStreamingResponse(StreamingResponse):
    """Release the synchronous model iterator, including after disconnection."""

    def __init__(self, iterator):
        self._source_iterator = iterator
        super().__init__(
            iterator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(_close_iterator, self._source_iterator)


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest, service=Depends(get_rag_service)):
    def events():
        iterator = None
        try:
            iterator = iter(service.ask_stream(payload.question, history=_history(payload)))
            for event in iterator:
                cleaned = _stream_event(event)
                yield _sse(cleaned)
                if cleaned["type"] == "done":
                    break
        except Exception as error:
            status, message = _safe_error(error, "流式问答")
            yield _sse({"type": "error", "message": message, "status_code": status})
        finally:
            if iterator is not None:
                _close_iterator(iterator)

    return ClosingStreamingResponse(events())


@app.get("/faqs", response_model=FAQListResponse)
def faq_list(service=Depends(get_rag_service)):
    try:
        items = [FAQItem(**item) for item in service.faq.list()]
        return FAQListResponse(items=items, count=len(items))
    except Exception as error:
        _raise_service_error(error, "FAQ 列表")


@app.post("/faqs", response_model=FAQItem)
def faq_add(payload: FAQCreate, service=Depends(get_rag_service)):
    try:
        item = service.faq.add(payload.question, payload.answer, aliases=payload.aliases)
        return FAQItem(**item)
    except Exception as error:
        _raise_service_error(error, "FAQ 添加")


@app.get("/faqs/settings", response_model=FAQSettingsResponse)
def faq_settings(service=Depends(get_rag_service)):
    try:
        return FAQSettingsResponse(**service.faq.get_settings())
    except Exception as error:
        _raise_service_error(error, "FAQ 匹配设置")


@app.put("/faqs/settings", response_model=FAQSettingsResponse)
def faq_settings_update(payload: FAQSettingsUpdate, service=Depends(get_rag_service)):
    try:
        updated = service.faq.update_settings(**payload.model_dump(exclude_none=True))
        return FAQSettingsResponse(**updated)
    except Exception as error:
        _raise_service_error(error, "FAQ 匹配设置更新")


@app.post("/faqs/match", response_model=FAQMatchResponse)
def faq_match(payload: FAQMatchRequest, service=Depends(get_rag_service)):
    try:
        return FAQMatchResponse(**service.faq.preview(payload.question))
    except Exception as error:
        _raise_service_error(error, "FAQ 匹配测试")


@app.delete("/faqs/{faq_id}")
def faq_delete(faq_id: str, service=Depends(get_rag_service)) -> dict[str, bool]:
    try:
        deleted = service.faq.delete(faq_id)
    except Exception as error:
        _raise_service_error(error, "FAQ 删除")
    if not deleted:
        raise HTTPException(status_code=404, detail="FAQ 不存在。")
    return {"deleted": True}


@app.post("/faqs/import", response_model=FAQImportResponse)
def faq_import(file: UploadFile = File(...), service=Depends(get_rag_service)):
    name = _filename(file.filename, "faqs.json")
    if Path(name).suffix.lower() != ".json":
        file.file.close()
        raise HTTPException(status_code=400, detail="FAQ 导入仅支持 JSON 文件。")
    content = _read_upload(file, max_bytes=MAX_FAQ_UPLOAD_BYTES)
    try:
        return FAQImportResponse(**service.faq.import_json(content))
    except Exception as error:
        _raise_service_error(error, "FAQ 导入")


@app.post("/audio/transcriptions", response_model=TranscriptionResponse)
def transcriptions(file: UploadFile = File(...), transcriber=Depends(get_transcriber)):
    name = _filename(file.filename, "recording.wav")
    mime_type = file.content_type or "audio/wav"
    content = _read_upload(file)
    try:
        text = transcriber.transcribe(content, filename=name, mime_type=mime_type)
        return TranscriptionResponse(text=text)
    except Exception as error:
        from speech import SpeechServiceError

        if isinstance(error, SpeechServiceError):
            logger.warning("语音服务请求失败：异常类型=%s", type(error).__name__)
            raise HTTPException(status_code=502, detail="语音服务暂时不可用，请稍后重试。") from None
        _raise_service_error(error, "语音转写")
