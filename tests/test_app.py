"""Streamlit 界面行为测试；模型及知识库服务使用本地替身。"""

from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch
import wave

import httpx
from langchain_core.documents import Document
from openai import APIConnectionError
from streamlit.runtime.uploaded_file_manager import UploadedFileRec
from streamlit.testing.v1 import AppTest
from speech import SpeechServiceError


APP = Path(__file__).resolve().parents[1] / "app.py"
CHAT_INPUT_KEY = "chat_composer"


class LocalFAQ:
    """只保存本次界面测试的问答，不访问用户数据或模型。"""

    def __init__(self):
        self.items = []
        self.settings = {"threshold": 0.70, "min_margin": 0.08}
        self.preview_calls = []

    def get_settings(self):
        return dict(self.settings)

    def update_settings(self, threshold, min_margin=None):
        self.settings["threshold"] = threshold
        if min_margin is not None:
            self.settings["min_margin"] = min_margin
        return self.get_settings()

    def preview(self, question):
        self.preview_calls.append(question)
        match = next((item for item in self.items if question in [item["question"], *item["aliases"]]), None)
        return {
            "question": question, "matched": match is not None,
            "reason": "exact" if match else "empty" if not self.items else "below_threshold",
            **self.get_settings(),
            "match": {**match, "match_method": "exact"} if match else None,
            "candidates": [],
        }

    def list(self):
        return deepcopy(self.items)

    def count(self):
        return len(self.items)

    def add(self, question, answer, aliases=None):
        question, answer = question.strip(), answer.strip()
        if not question or not answer:
            raise ValueError("FAQ 标准问题和标准答案不能为空。")
        item = {"id": str(len(self.items) + 1), "question": question,
                "answer": answer, "aliases": list(aliases or [])}
        self.items.append(item)
        return deepcopy(item)

    def delete(self, item_id):
        before = self.count()
        self.items = [item for item in self.items if item["id"] != item_id]
        return before != self.count()

    def add_aliases(self, item_id, aliases):
        item = next(item for item in self.items if item["id"] == item_id)
        for alias in aliases:
            if alias not in [item["question"], *item["aliases"]]:
                item["aliases"].append(alias)
        return deepcopy(item)

    def import_json(self, content):
        added = skipped = 0
        for item in json.loads(content.decode("utf-8-sig")):
            if any(existing["question"] == item["question"] for existing in self.items):
                skipped += 1
            else:
                self.add(item["question"], item["answer"], item.get("aliases"))
                added += 1
        return {"added": added, "skipped": skipped, "count": self.count()}

    def match(self, question):
        for item in self.items:
            if question in [item["question"], *item["aliases"]]:
                return Document(page_content=item["answer"], metadata={
                    "kind": "faq", "faq_id": item["id"],
                    "source": "FAQ：" + item["question"], "relevance_score": 1.0,
                })
        return None


class LocalService:
    def __init__(self, settings):
        self.settings = settings
        self.calls = []
        self.faq = LocalFAQ()

    def count(self):
        return 1

    def ask(self, question, history=None):
        match = self.faq.match(question)
        if match is not None:
            return match.page_content, [match]
        self.calls.append({"question": question, "history": deepcopy(history or [])})
        if not self.count():
            return "我们可以直接聊天。", []
        return "活动在周六举行。[1]", [
            Document(
                page_content="活动在周六举行。",
                metadata={"source": "说明.md", "relevance_score": 0.9},
            )
        ]

    def ask_stream(self, question, history=None):
        answer, documents = self.ask(question, history)
        sources = [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in documents
        ]
        mode = "faq" if documents and documents[0].metadata.get("kind") == "faq" else "knowledge" if sources else "model"
        yield {"type": "sources", "sources": sources, "mode": mode}
        midpoint = len(answer) // 2
        yield {"type": "token", "content": answer[:midpoint]}
        yield {"type": "token", "content": answer[midpoint:]}
        yield {"type": "done", "answer": answer}


class AppTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(
            os.environ,
            {"DASHSCOPE_API_KEY": "", "DASHSCOPE_BASE_URL": "", "CHAT_MODEL": "",
             "EMBEDDING_MODEL": ""},
        )
        environment.start()
        self.addCleanup(environment.stop)
        # 这些测试不读取用户的本地凭据，也不访问实际知识库。
        dotenv = patch("dotenv.load_dotenv")
        dotenv.start()
        self.addCleanup(dotenv.stop)
        service = patch("rag.RAGService", LocalService)
        service.start()
        self.addCleanup(service.stop)

    def configured_app(self):
        os.environ.update(
            DASHSCOPE_BASE_URL="https://example.invalid/compatible-mode/v1",
            CHAT_MODEL="qwen-plus",
            EMBEDDING_MODEL="text-embedding-v4",
        )
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        app.text_input[0].set_value("test-key-only").run()
        self.assertEqual(len(app.exception), 0)
        return app

    def faq_app(self):
        return self.configured_app().switch_page("views/faq.py").run()

    def isolated_faq_app(self):
        def faq_test_page():
            import streamlit as st
            from faq_page_ui import render_faq_page
            from test_app import LocalService

            if "service" not in st.session_state:
                st.session_state["service"] = LocalService(None)

            def show_error(error, operation=""):
                st.error(f"{operation}：{error}")

            render_faq_page(st.session_state["service"], show_error)

        app = AppTest.from_function(faq_test_page, default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def submit_audio(self, app, text=""):
        """模拟前端上传并提交录音，走真实 WidgetState 与上传文件反序列化。"""
        wav = BytesIO()
        with wave.open(wav, "wb") as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(16000)
            recording.writeframes(b"\x00\x00" * 1600)
        audio_bytes = wav.getvalue()
        file_record = UploadedFileRec(
            file_id="chat-microphone-recording",
            name="recording.wav",
            type="audio/wav",
            data=audio_bytes,
        )
        widget_states = app._tree.get_widget_states()
        chat_widget = next(
            widget for widget in widget_states.widgets
            if widget.id == app.chat_input[0].id
        )
        chat_widget.chat_input_value.data = text
        audio_info = chat_widget.chat_input_value.audio_file_info
        audio_info.file_id = file_record.file_id
        audio_info.name = file_record.name
        audio_info.size = len(audio_bytes)
        original_register = app._register_uploaded_files

        def register_with_recording(script_runner):
            original_register(script_runner)
            script_runner.register_file(file_record)

        with patch.object(app, "_register_uploaded_files", register_with_recording):
            app._run(widget_states)
        self.assertEqual(len(app.exception), 0)
        return audio_bytes

    def test_missing_key_shows_setup_without_crashing(self):
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("API Key" in info.value for info in app.info))
        self.assertEqual(len(app.chat_input), 0)

    def test_web_uses_shared_environment_defaults_and_keeps_session_overrides_local(self):
        from settings import load_rag_settings

        os.environ.update(
            DASHSCOPE_API_KEY="test-key-only", RAG_TOP_K="7", RAG_RELEVANCE_THRESHOLD="0.6",
        )
        app = self.configured_app()
        self.assertEqual(app.session_state["settings"], load_rag_settings())
        self.assertEqual(app.slider[0].value, 0.6)
        app.slider[0].set_value(0.65).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["settings"].relevance_threshold, 0.65)
        self.assertEqual(app.session_state["settings"].top_k, 7)
        self.assertEqual(load_rag_settings().relevance_threshold, 0.6)

    def test_answer_sources_and_clear_conversation(self):
        app = self.configured_app()
        app.chat_input[0].set_value("活动什么时候举行？").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.chat_message), 2)
        self.assertIn("周六", app.chat_message[1].markdown[0].value)
        self.assertEqual(app.chat_message[1].caption[0].value, "本地知识参考 · 模型补充")
        self.assertTrue(any("说明.md" in element.value for element in app.text))
        next(button for button in app.button if button.label == "清空对话记录").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.chat_message), 0)
        self.assertEqual(app.metric[0].value, "1")

    def test_changing_embedding_model_resets_visible_conversation(self):
        app = self.configured_app()
        app.chat_input[0].set_value("活动什么时候举行？").run()
        app.text_input[3].set_value("another-embedding-model").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.chat_message), 0)
        self.assertEqual(app.session_state["settings"].embedding_model, "another-embedding-model")

    def test_empty_knowledge_base_allows_normal_chat_without_fake_sources(self):
        with patch.object(LocalService, "count", return_value=0):
            app = self.configured_app()
            self.assertEqual(len(app.chat_input), 1)
            self.assertFalse(app.chat_input[0].disabled)
            self.assertEqual(app.chat_input[0].placeholder, "聊点什么，或问问本地资料…")
            self.assertTrue(
                any(
                    info.value == "还没有本地资料，可以直接聊天；上传文档后会优先参考相关内容。"
                    for info in app.info
                )
            )
            self.assertEqual(app.metric[0].value, "0")
            app.chat_input[0].set_value("你好，可以聊聊吗？").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.chat_message), 2)
        self.assertEqual(app.chat_message[1].caption[0].value, "模型回答")
        self.assertIn("可以直接聊天", app.chat_message[1].markdown[0].value)
        self.assertEqual(app.session_state["messages"][-1]["sources"], [])
        self.assertFalse(any(expander.label == "查看参考原文" for expander in app.expander))

    def test_second_question_receives_previous_user_and_assistant_messages(self):
        app = self.configured_app()
        first_question = "活动什么时候举行？"
        app.chat_input[0].set_value(first_question).run()
        service = app.session_state["service"]
        self.assertEqual(service.calls[0], {"question": first_question, "history": []})
        app.chat_input[0].set_value("那是在周几？").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(service.calls), 2)
        self.assertEqual(service.calls[1]["question"], "那是在周几？")
        history = service.calls[1]["history"]
        self.assertEqual([message["role"] for message in history], ["user", "assistant"])
        self.assertEqual(history[0]["content"], first_question)
        self.assertIn("周六", history[1]["content"])
        self.assertEqual(len(app.chat_message), 4)

    def test_threshold_and_chat_model_changes_preserve_conversation(self):
        app = self.configured_app()
        app.chat_input[0].set_value("活动什么时候举行？").run()
        messages = deepcopy(app.session_state["messages"])
        app.slider[0].set_value(0.55).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["messages"], messages)
        self.assertAlmostEqual(app.session_state["settings"].relevance_threshold, 0.55)
        app.text_input[2].set_value("qwen-turbo").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["messages"], messages)
        self.assertEqual(app.session_state["settings"].chat_model, "qwen-turbo")

    def test_prompt_update_replaces_cached_service_and_preserves_conversation(self):
        from prompts import PROMPT_VERSION

        app = self.configured_app()
        app.chat_input[0].set_value("活动什么时候举行？").run()
        previous_service = app.session_state["service"]
        messages = deepcopy(app.session_state["messages"])
        app.session_state["prompt_version"] = "previous-prompt"
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertIsNot(app.session_state["service"], previous_service)
        self.assertEqual(app.session_state["messages"], messages)
        self.assertEqual(app.session_state["prompt_version"], PROMPT_VERSION)

    def test_upload_connection_error_has_context_without_leaking_secret(self):
        namespace = runpy.run_path(str(APP), run_name="app_error_test")
        secret = "SECRET-MUST-NOT-APPEAR-IN-DIAGNOSTICS"
        error = APIConnectionError(
            message=f"Mock connection failed with {secret}",
            request=httpx.Request("POST", "https://example.invalid/embeddings"),
        )
        error.__cause__ = ConnectionError(10054, f"Mock socket error with {secret}")
        with (
            patch("streamlit.error") as display_error,
            patch("logging.Logger._log") as diagnostic_log,
        ):
            namespace["show_error"](error, operation="文档加入失败")

        self.assertTrue(display_error.called)
        messages = " ".join(str(call.args[0]) for call in display_error.call_args_list)
        self.assertIn("文档加入失败", messages)
        self.assertIn("无法连接百炼", messages)
        self.assertNotIn(secret, messages)
        self.assertTrue(diagnostic_log.called)
        diagnostics = repr(diagnostic_log.call_args_list)
        self.assertIn("APIConnectionError", diagnostics)
        self.assertIn("ConnectionError", diagnostics)
        self.assertIn("10054", diagnostics)
        self.assertNotIn(secret, diagnostics)

    def test_stream_failure_does_not_save_partial_answer(self):
        def broken_stream(_service, question, history=None):
            yield {"type": "sources", "sources": [], "mode": "model"}
            yield {"type": "token", "content": "尚未完整生成的回答"}
            raise RuntimeError("SECRET-STREAM-ERROR")

        with patch.object(LocalService, "ask_stream", broken_stream):
            app = self.configured_app()
            app.chat_input[0].set_value("你好").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["messages"], [])
        self.assertTrue(any("回答失败" in error.value for error in app.error))
        self.assertFalse(any("SECRET-STREAM-ERROR" in error.value for error in app.error))

    def test_microphone_is_in_the_chat_composer_without_separate_voice_controls(self):
        app = self.configured_app()
        self.assertEqual(app.chat_input[0].key, CHAT_INPUT_KEY)
        self.assertTrue(app.chat_input[0].proto.accept_audio)
        self.assertEqual(app.chat_input[0].proto.audio_sample_rate, 16000)
        self.assertEqual(app.chat_input[0].proto.max_chars, 0)
        self.assertEqual(len(app.audio_input), 0)
        self.assertEqual(len(app.text_area), 0)
        self.assertFalse(any("语音输入" in expander.label for expander in app.expander))
        self.assertFalse(any(
            button.label in {"识别为文字", "发送识别文字"} for button in app.button
        ))

    def test_stopping_recording_fills_chat_composer_without_sending(self):
        with patch("speech.create_transcriber") as factory:
            factory.return_value.transcribe.return_value = "请介绍机器学习"
            app = self.configured_app()
            audio_bytes = self.submit_audio(app)
            self.assertTrue(app.chat_input[0].proto.set_value)
            self.assertEqual(app.chat_input[0].proto.value, "请介绍机器学习")
            self.assertEqual(app.session_state["service"].calls, [])
            self.assertEqual(app.session_state["messages"], [])
            self.assertEqual(len(app.chat_message), 0)
            factory.return_value.transcribe.assert_called_once_with(
                audio_bytes, filename="recording.wav", mime_type="audio/wav"
            )
            factory.assert_called_once_with(
                fallback_api_key="test-key-only",
                fallback_base_url="https://example.invalid/compatible-mode/v1",
            )

    def test_voice_transcription_can_be_edited_and_sent_once(self):
        with patch("speech.create_transcriber") as factory:
            factory.return_value.transcribe.return_value = "请介绍机器学习"
            app = self.configured_app()
            self.submit_audio(app)
            app.chat_input[0].set_value("请给我一个机器学习例子").run()
            self.assertEqual(len(app.exception), 0)
            service = app.session_state["service"]
            self.assertEqual(service.calls[-1]["question"], "请给我一个机器学习例子")
            self.assertEqual(len(service.calls), 1)
            app.run()
            self.assertEqual(len(service.calls), 1)
            self.assertEqual(len(app.chat_message), 2)
            factory.return_value.transcribe.assert_called_once()

    def test_rerender_does_not_repeat_transcription_or_send_recorded_draft(self):
        with patch("speech.create_transcriber") as factory:
            factory.return_value.transcribe.return_value = "等待用户确认的文字"
            app = self.configured_app()
            self.submit_audio(app)
            app.run()
            app.run()
            app.slider[0].set_value(0.55).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["service"].calls, [])
            self.assertEqual(len(app.chat_message), 0)
            factory.return_value.transcribe.assert_called_once()

    def test_recording_preserves_text_already_in_the_composer(self):
        with patch("speech.create_transcriber") as factory:
            factory.return_value.transcribe.return_value = "再加一个实际例子"
            app = self.configured_app()
            self.submit_audio(app, text="先解释概念")
            self.assertEqual(app.chat_input[0].proto.value, "先解释概念 再加一个实际例子")
            self.assertEqual(app.session_state["service"].calls, [])
            app.chat_input[0].set_value("先解释概念 再加一个实际例子").run()
            self.assertEqual(len(app.session_state["service"].calls), 1)

    def test_voice_failure_preserves_original_draft_and_keeps_text_chat_available(self):
        with patch("speech.create_transcriber") as factory:
            factory.return_value.transcribe.side_effect = SpeechServiceError("语音服务暂时不可用。")
            app = self.configured_app()
            self.submit_audio(app, text="还没写完的草稿")
            self.assertTrue(app.chat_input[0].proto.set_value)
            self.assertEqual(app.chat_input[0].proto.value, "还没写完的草稿")
            self.assertEqual(app.session_state["service"].calls, [])
            self.assertTrue(any("语音识别失败" in error.value for error in app.error))
            self.assertFalse(app.chat_input[0].disabled)
            app.run()
            factory.return_value.transcribe.assert_called_once()
            app.chat_input[0].set_value("改为文字提问").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["service"].calls[-1]["question"], "改为文字提问")

    def test_long_voice_transcript_remains_editable_before_sending(self):
        transcript = "测试" * 2001
        with patch("speech.create_transcriber") as factory:
            factory.return_value.transcribe.return_value = transcript
            app = self.configured_app()
            self.submit_audio(app)
            self.assertEqual(app.chat_input[0].proto.value, transcript)
            self.assertEqual(app.chat_input[0].proto.max_chars, 0)
            self.assertEqual(app.session_state["service"].calls, [])
            app.chat_input[0].set_value(transcript).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.chat_input[0].proto.value, transcript)
            self.assertEqual(app.session_state["service"].calls, [])
            self.assertTrue(any("4000" in info.value for info in app.info))
            app.chat_input[0].set_value("删减后可以发送").run()
            self.assertEqual(len(app.session_state["service"].calls), 1)

    def test_typing_a_long_message_preserves_it_for_editing_without_model_call(self):
        draft = "长" * 4001
        app = self.configured_app()
        app.chat_input[0].set_value(draft).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.chat_input[0].proto.set_value)
        self.assertEqual(app.chat_input[0].proto.value, draft)
        self.assertEqual(app.session_state["service"].calls, [])
        self.assertTrue(any("4000" in info.value for info in app.info))

    def test_adding_faq_returns_standard_answer_and_keeps_normal_chat_available(self):
        app = self.faq_app()
        next(widget for widget in app.text_input if widget.label == "标准问题").set_value("如何退货？")
        next(widget for widget in app.text_area if widget.label == "标准答案").set_value("请在订单页申请售后。")
        next(widget for widget in app.text_area if "相似问法" in widget.label).set_value("  我想退货  \n\n怎么退货\n")
        next(button for button in app.button if button.label == "添加 FAQ").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["service"].faq.count(), 1)
        self.assertEqual(app.session_state["service"].faq.list()[0]["aliases"], ["我想退货", "怎么退货"])
        app.switch_page("views/chat.py").run()
        app.chat_input[0].set_value("我想退货").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["service"].calls, [])
        self.assertEqual(app.chat_message[1].markdown[0].value, "请在订单页申请售后。")
        self.assertEqual(app.chat_message[1].caption[0].value, "FAQ 标准答案")
        self.assertTrue(any("FAQ：如何退货？" in element.value for element in app.text))
        app.chat_input[0].set_value("活动什么时候举行？").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state["service"].calls), 1)
        self.assertEqual(app.chat_message[1].caption[0].value, "FAQ 标准答案")
        self.assertEqual(app.chat_message[3].caption[0].value, "本地知识参考 · 模型补充")

    def test_loading_demo_twice_skips_duplicates_and_deleting_removes_entry(self):
        app = self.faq_app()
        next(button for button in app.button if button.label == "加载客服示例").click().run()
        self.assertEqual(len(app.exception), 0)
        service = app.session_state["service"]
        count = service.faq.count()
        self.assertGreater(count, 0)
        next(button for button in app.button if button.label == "加载客服示例").click().run()
        self.assertEqual(service.faq.count(), count)
        self.assertTrue(any("新增 0 条" in item.value for item in app.success))
        first = service.faq.list()[0]
        next(button for button in app.button if button.key == f"delete_faq_{first['id']}").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(service.faq.count(), count - 1)
        self.assertNotIn(first["id"], [item["id"] for item in service.faq.list()])
        app.switch_page("views/chat.py").run()
        app.chat_input[0].set_value(first["question"]).run()
        self.assertEqual(service.calls[-1]["question"], first["question"])

    def test_importing_json_faq_makes_it_available_in_chat(self):
        app = self.faq_app()
        content = json.dumps([{"question": "如何查询物流？", "answer": "请在订单页查看物流。"}], ensure_ascii=False).encode()
        app.file_uploader(key="faq_upload").set_value(("faq.json", content, "application/json")).run()
        next(button for button in app.button if button.label == "导入 FAQ").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["service"].faq.count(), 1)
        app.switch_page("views/chat.py").run()
        app.chat_input[0].set_value("如何查询物流？").run()
        self.assertEqual(app.chat_message[1].caption[0].value, "FAQ 标准答案")
        self.assertEqual(app.chat_message[1].markdown[0].value, "请在订单页查看物流。")

    def test_adding_aliases_preserves_faq_answer_id_history_and_matches_new_question(self):
        app = self.faq_app()
        service = app.session_state["service"]
        item = service.faq.add("如何申请退款？", "请在订单页点击申请售后。", ["怎么退款"])
        messages = [{"role": "user", "content": "已有对话"}]
        app.session_state["messages"] = deepcopy(messages)
        app.run()
        app.text_area(key=f"faq_aliases_{item['id']}").set_value("  退款在哪里  \n\n怎么退款\n")
        next(button for button in app.button if button.key == f"save_faq_aliases_{item['id']}").click().run()
        self.assertEqual(len(app.exception), 0)
        saved = service.faq.list()[0]
        self.assertEqual(saved["id"], item["id"])
        self.assertEqual(saved["answer"], item["answer"])
        self.assertEqual(saved["aliases"], ["怎么退款", "退款在哪里"])
        self.assertEqual(app.session_state["messages"], messages)
        app.switch_page("views/chat.py").run()
        app.chat_input[0].set_value("退款在哪里").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(service.calls, [])
        self.assertEqual(app.chat_message[-1].caption[0].value, "FAQ 标准答案")
        self.assertEqual(app.chat_message[-1].markdown[0].value, item["answer"])

    def test_empty_faq_form_reports_error_without_disabling_chat(self):
        app = self.faq_app()
        next(button for button in app.button if button.label == "添加 FAQ").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["service"].faq.count(), 0)
        self.assertTrue(any("FAQ 保存失败" in error.value for error in app.error))
        app.switch_page("views/chat.py").run()
        self.assertFalse(app.chat_input[0].disabled)

    def test_chat_page_does_not_contain_faq_management_controls(self):
        app = self.configured_app()
        self.assertFalse(any(widget.label == "标准问题" for widget in app.text_input))
        self.assertEqual(len(app.text_area), 0)
        self.assertFalse(any(button.label == "添加 FAQ" for button in app.button))
        self.assertFalse(any(expander.label == "客服 FAQ" for expander in app.expander))

    def test_faq_page_has_management_settings_and_preview_tabs_in_main_area(self):
        app = self.isolated_faq_app()
        self.assertEqual([tab.label for tab in app.tabs], ["问答管理", "匹配设置", "匹配测试"])
        self.assertEqual(len(app.sidebar.text_input), 0)
        self.assertEqual(len(app.sidebar.text_area), 0)
        self.assertEqual(len(app.chat_input), 0)
        settings_input = next(widget for widget in app.number_input if "匹配门槛" in widget.label)
        self.assertEqual(settings_input.value, 70)
        self.assertEqual(settings_input.min, 0)
        self.assertEqual(settings_input.max, 100)
        self.assertTrue(any("不能绕过语义门槛" in info.value for info in app.info))

    def test_saving_faq_settings_preserves_service_and_conversation_across_pages(self):
        app = self.configured_app()
        app.chat_input[0].set_value("活动什么时候举行？").run()
        service = app.session_state["service"]
        messages = deepcopy(app.session_state["messages"])
        app.switch_page("views/faq.py").run()
        next(widget for widget in app.number_input if "匹配门槛" in widget.label).set_value(80)
        next(widget for widget in app.number_input if "候选领先分" in widget.label).set_value(12)
        next(button for button in app.button if button.label == "保存匹配设置").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIs(app.session_state["service"], service)
        self.assertEqual(service.faq.get_settings(), {"threshold": 0.8, "min_margin": 0.12})
        self.assertEqual(app.session_state["messages"], messages)
        self.assertEqual(len(service.calls), 1)
        app.switch_page("views/chat.py").run()
        self.assertEqual(app.session_state["messages"], messages)
        self.assertEqual(len(app.chat_message), 2)

    def test_match_preview_does_not_call_answer_model_or_change_chat_history(self):
        app = self.isolated_faq_app()
        service = app.session_state["service"]
        service.faq.add("如何退货？", "请在订单页申请售后。", ["怎么退货"])
        messages = [{"role": "user", "content": "之前的问题"}]
        app.session_state["messages"] = deepcopy(messages)
        next(widget for widget in app.text_input if widget.label == "测试问题").set_value("怎么退货")
        next(button for button in app.button if button.label == "测试匹配").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("命中 FAQ（精确匹配）" in success.value for success in app.success))
        self.assertEqual(service.faq.preview_calls, ["怎么退货"])
        self.assertEqual(service.calls, [])
        self.assertEqual(app.session_state["messages"], messages)

    def test_preview_shows_fallback_and_semantic_points_separately_from_raw_bm25(self):
        preview = {
            "question": "怎么申请", "matched": False, "reason": "below_threshold",
            "threshold": 0.7, "min_margin": 0.08, "match": None,
            "candidates": [{"id": "1", "question": "如何退货？",
                            "semantic_score": 0.69, "bm25_score": 2.34567,
                            "hybrid_score": 0.74}],
        }
        app = self.isolated_faq_app()
        with patch.object(LocalFAQ, "preview", return_value=preview) as called:
            next(widget for widget in app.text_input if widget.label == "测试问题").set_value("怎么申请")
            next(button for button in app.button if button.label == "测试匹配").click().run()
        called.assert_called_once_with("怎么申请")
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("未命中 FAQ，将继续 RAG" in info.value for info in app.info))
        table = app.dataframe[0].value
        self.assertEqual(table.iloc[0]["语义分（0–100）"], 69.0)
        self.assertEqual(table.iloc[0]["混合分（0–100）"], 74.0)
        self.assertAlmostEqual(table.iloc[0]["BM25 原始分"], 2.3457)
        self.assertEqual(app.session_state["service"].calls, [])


if __name__ == "__main__":
    unittest.main()
