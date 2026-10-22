"""运行：python -m streamlit run app.py。"""

from dataclasses import replace
import logging

import streamlit as st
from openai import APIConnectionError, APIStatusError, APITimeoutError

from prompts import PROMPT_VERSION
from rag import RAGService
from speech import SpeechServiceError
from settings import load_rag_settings


logger = logging.getLogger(__name__)
CHAT_INPUT_KEY = "chat_composer"


def show_error(error: Exception, operation: str = "") -> None:
    """显示可操作的提示，避免把带认证信息的底层异常直接输出。"""
    if isinstance(error, SpeechServiceError):
        message = str(error)
    elif isinstance(error, APITimeoutError):
        message = "百炼请求超时，请稍后重试。"
    elif isinstance(error, APIConnectionError):
        # 只记录异常类型和系统错误码，原始异常可能包含认证信息。
        types, codes, seen = [], set(), set()
        cause = error
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            types.append(type(cause).__name__)
            for attribute in ("winerror", "errno"):
                code = getattr(cause, attribute, None)
                if isinstance(code, int):
                    codes.add(code)
            cause = cause.__cause__ or cause.__context__
        logger.warning("百炼连接失败：阶段=%s，异常类型=%s，系统错误码=%s", operation, types, sorted(codes))
        if 10013 in codes:
            message = "应用的网络访问被系统阻止，请在本机终端重新启动应用后再试。"
        else:
            message = "无法连接百炼，请检查网络和调用地址。"
    elif isinstance(error, APIStatusError):
        messages = {
            400: "百炼未接受请求，请检查模型名称和输入是否受支持。",
            401: "API Key 无效，请检查 Key 及其所属地域。",
            403: "没有调用权限，请检查模型开通状态及 Key 所属地域。",
            404: "未找到接口或模型，请检查调用地址和模型名称。",
            429: "请求过于频繁或额度不足，请稍后重试并检查百炼控制台。",
        }
        message = messages.get(error.status_code, "百炼服务暂时不可用，请稍后重试。")
    elif type(error) is ValueError:
        message = str(error)
    else:
        message = "处理失败，请检查文件、配置和本地数据目录后重试。"
    st.error(f"{operation}：{message}" if operation else message)


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("查看参考原文"):
        for index, source in enumerate(sources, start=1):
            metadata = source["metadata"]
            label = f"[{index}] {metadata.get('source', '文档')}"
            if "page" in metadata:
                label += f" · 第 {metadata['page']} 页"
            if metadata.get("kind") == "faq" and metadata.get("match_method") == "exact":
                label += " · 精确匹配"
            elif "relevance_score" in metadata:
                label += f" · 相关度 {metadata['relevance_score']:.2f}"
            st.text(label)
            st.text(source["content"])


def answer_caption(sources: list[dict]) -> str:
    if sources and sources[0].get("metadata", {}).get("kind") == "faq":
        return "FAQ 标准答案"
    return "本地知识参考 · 模型补充" if sources else "模型回答"


def write_streamed_answer(service, question: str, history: list[dict]) -> tuple[str, list[dict]]:
    """逐段展示真实模型输出；未完整结束的回答不写入对话历史。"""
    state = {"sources": [], "done": False}
    status = st.empty()
    status.caption("正在查找资料并思考…")

    def chunks():
        events = iter(service.ask_stream(question, history=history))
        try:
            for event in events:
                if event["type"] == "sources":
                    state["sources"] = event["sources"]
                    status.caption(answer_caption(event["sources"]))
                elif event["type"] == "token":
                    yield event["content"]
                elif event["type"] == "done":
                    state["done"] = True
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()

    try:
        answer = st.write_stream(chunks())
        if not state["done"] or not isinstance(answer, str) or not answer.strip():
            raise ValueError("回答未完整生成，请重新发送问题。")
    except Exception:
        status.caption("回答中断，请重新发送问题。")
        raise
    return answer, state["sources"]


def queue_audio_transcription() -> None:
    """录音完成只安排转写，并恢复原草稿；绝不直接作为问题发送。"""
    submission = st.session_state.get(CHAT_INPUT_KEY)
    if submission is None:
        return
    text = submission if isinstance(submission, str) else submission.text
    audio = None if isinstance(submission, str) else submission.audio
    if audio is not None:
        st.session_state["pending_audio_transcription"] = {
            "audio": audio.getvalue(),
            "filename": audio.name,
            "mime_type": audio.type,
            "text": text,
        }
        # 原生录音完成会提交音频并清空输入框。赋纯字符串仅填回草稿，
        # st.chat_input 本轮返回 None，避免识别前后自动触发聊天。
        st.session_state[CHAT_INPUT_KEY] = text
    elif len(text.strip()) > 4000:
        st.session_state[CHAT_INPUT_KEY] = text
        st.session_state["composer_notice"] = "单次消息最多 4000 个字符，请在输入栏中删减后发送。"


def transcribe_pending_audio(api_key: str, base_url: str) -> None:
    """在输入框创建前回填文字；音频从队列取出一次，重绘不重复识别。"""
    pending = st.session_state.pop("pending_audio_transcription", None)
    if pending is None:
        return
    try:
        from speech import create_transcriber

        transcriber = create_transcriber(
            fallback_api_key=api_key, fallback_base_url=base_url
        )
        with st.spinner("正在识别语音，完成后会填入输入栏…"):
            text = transcriber.transcribe(
                pending["audio"],
                filename=pending["filename"],
                mime_type=pending["mime_type"],
            )
        draft = pending["text"]
        st.session_state[CHAT_INPUT_KEY] = f"{draft.rstrip()} {text}" if draft.strip() else text
        if len(st.session_state[CHAT_INPUT_KEY].strip()) > 4000:
            st.session_state["composer_notice"] = "单次消息最多 4000 个字符，请在输入栏中删减后发送。"
    except Exception as error:
        # callback 已恢复录音前的草稿，失败也不影响继续打字。
        show_error(error, "语音识别失败")


def connection_service():
    """入口共用的连接控件；切换页面时保留连接和聊天状态。"""
    try:
        defaults = load_rag_settings()
    except ValueError as error:
        show_error(error, "配置加载失败")
        return None
    with st.sidebar:
        st.header("1. 连接百炼")
        api_key = st.text_input(
            "API Key",
            value=defaults.api_key,
            type="password",
            help="可在本地 .env 配置，也可在此临时输入；网页输入不会写入文件。",
        ).strip()
        with st.expander("模型与调用地址"):
            base_url = st.text_input(
                "Base URL",
                value=defaults.base_url,
                help="填写与 API Key 同地域的 OpenAI 兼容地址，以 /compatible-mode/v1 结尾。",
            ).strip().rstrip("/")
            chat_model = st.text_input(
                "回答模型", value=defaults.chat_model
            ).strip()
            embedding_model = st.text_input(
                "向量模型", value=defaults.embedding_model
            ).strip()
            relevance_threshold = st.slider(
                "最低资料相关度",
                min_value=0.0,
                max_value=1.0,
                value=defaults.relevance_threshold,
                step=0.05,
                help="越高越严格。仅引用达到此值的资料，其余问题由模型回答。相关度采用余弦相似度，不代表回答正确率。",
            )
            st.caption("更换向量模型或调用地址后，会使用独立知识库，需要重新上传文档。")

    if not api_key:
        st.info("先在左侧填写百炼 API Key，或配置项目中的 .env 文件。")
        st.markdown("连接后即可聊天，也可以上传文档，让助手参考你的本地资料。")
        return None

    try:
        settings = replace(
            defaults,
            api_key=api_key,
            base_url=base_url,
            chat_model=chat_model,
            embedding_model=embedding_model,
            relevance_threshold=relevance_threshold,
        )
        previous_settings = st.session_state.get("settings")
        if (
            previous_settings != settings
            or st.session_state.get("prompt_version") != PROMPT_VERSION
        ):
            service = RAGService(settings)
            st.session_state["service"] = service
            st.session_state["settings"] = settings
            st.session_state["prompt_version"] = PROMPT_VERSION
            # 调整相关度或回答模型保留对话；切换知识库时避免混用旧资料。
            if previous_settings is None or (
                previous_settings.base_url, previous_settings.embedding_model
            ) != (settings.base_url, settings.embedding_model):
                st.session_state["messages"] = []
        service = st.session_state["service"]
    except Exception as error:
        show_error(error)
        return None
    return service


def render_chat_page() -> None:
    st.title("💬 我的智能助手")
    st.caption("随时聊天，优先参考 FAQ 与本地知识。")
    service = st.session_state["service"]
    settings = st.session_state["settings"]
    api_key, base_url = settings.api_key, settings.base_url
    with st.sidebar:
        st.header("本地知识（可选）")
        uploads = st.file_uploader(
            "选择 TXT、Markdown 或文字型 PDF",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
            help="每个文件最多 20 MB。扫描件暂不支持。",
        )
        add_clicked = st.button("加入知识库", type="primary", disabled=not uploads)
        st.caption("文档片段和向量保存在本机；建库和问答会将必要文本发送到百炼。")
        if st.button("清空对话记录"):
            st.session_state["messages"] = []

    if add_clicked:
        for upload in uploads:
            try:
                if upload.size > 20 * 1024 * 1024:
                    raise ValueError("文件超过 20 MB，请拆分后再上传。")
                with st.spinner(f"正在处理 {upload.name}…"):
                    added = service.add_file(upload.name, upload.getvalue())
                if added:
                    st.sidebar.success(f"{upload.name}：新增 {added} 段")
                else:
                    st.sidebar.info(f"{upload.name}：已在知识库中，跳过重复上传。")
            except Exception as error:
                st.sidebar.warning(f"未完成：{upload.name}。可修正后重新加入。")
                show_error(error, "文档加入失败")

    try:
        count = service.count()
    except Exception as error:
        show_error(error, "知识库暂不可用")
        count = 0
    st.sidebar.metric("知识库文本段数", count)
    st.subheader("开始对话")
    st.caption("优先匹配 FAQ，再参考本地资料，其他问题由模型回答。支持连续追问；尚未接入实时联网查询。")
    if not count:
        st.info("还没有本地资料，可以直接聊天；上传文档后会优先参考相关内容。")

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.caption(answer_caption(message.get("sources", [])))
            st.markdown(message["content"])
            show_sources(message.get("sources", []))

    transcribe_pending_audio(api_key, base_url)
    notice = st.session_state.pop("composer_notice", None)
    if notice:
        st.info(notice)
    submission = st.chat_input(
        "聊点什么，或问问本地资料…",
        key=CHAT_INPUT_KEY,
        accept_audio=True,
        audio_sample_rate=16000,
        max_upload_size=10,
        submit_mode="disable",
        on_submit=queue_audio_transcription,
    )
    question = submission.text.strip() if submission is not None else None
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.chat_message("assistant"):
            answer, sources = write_streamed_answer(
                service, question, st.session_state.get("messages", [])
            )
            show_sources(sources)
        st.session_state.setdefault("messages", []).extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer, "sources": sources},
            ]
        )
    except Exception as error:
        show_error(error, "回答失败")


def main() -> None:
    st.set_page_config(page_title="我的智能助手", page_icon="💬", layout="centered")
    page = st.navigation(
        [
            st.Page("views/chat.py", title="聊天", icon="💬", default=True),
            st.Page("views/faq.py", title="FAQ 管理", icon="📋", url_path="faq"),
        ],
        position="top",
    )
    if connection_service() is not None:
        page.run()


if __name__ == "__main__":
    main()
