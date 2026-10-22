"""Offline HTTP checks; dependency overrides never use credentials or a cloud API."""

import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx
from langchain_core.documents import Document
from openai import APIConnectionError, APIStatusError, APITimeoutError
from starlette.requests import ClientDisconnect

import api


class FakeFAQ:
    def __init__(self):
        self.items = []
        self.calls = []
        self.settings = {"threshold": 0.7, "min_margin": 0.05}
        self.preview_result = {
            "question": "如何退款？",
            "matched": True,
            "reason": "hybrid",
            "threshold": 0.7,
            "min_margin": 0.05,
            "match": {
                "id": "faq-1", "question": "如何退款？", "answer": "联系客服办理。", "match_method": "semantic"
            },
            "candidates": [
                {"id": "faq-1", "question": "如何退款？", "semantic_score": 0.82, "bm25_score": 2.5, "hybrid_score": 0.88}
            ],
        }

    def list(self):
        return deepcopy(self.items)

    def count(self):
        return len(self.items)

    def add(self, question, answer, aliases=None):
        self.calls.append(("add", question, answer, deepcopy(aliases)))
        item = {
            "id": f"faq-{len(self.items) + 1}",
            "question": question,
            "answer": answer,
            "aliases": list(aliases or []),
            "updated_at": "2026-10-16T00:00:00+00:00",
        }
        self.items.append(item)
        return deepcopy(item)

    def delete(self, faq_id):
        self.calls.append(("delete", faq_id))
        for index, item in enumerate(self.items):
            if item["id"] == faq_id:
                self.items.pop(index)
                return True
        return False

    def import_json(self, content):
        self.calls.append(("import", content))
        return {"added": 2, "skipped": 1, "count": self.count() + 2}

    def get_settings(self):
        return deepcopy(self.settings)

    def update_settings(self, **settings):
        self.calls.append(("settings", deepcopy(settings)))
        self.settings.update(settings)
        return self.get_settings()

    def preview(self, question):
        self.calls.append(("preview", question))
        result = deepcopy(self.preview_result)
        result["question"] = question
        return result


class FakeService:
    def __init__(self):
        self.faq = FakeFAQ()
        self.chunk_count = 3
        self.uploads = []
        self.calls = []
        self.documents = [
            Document(
                page_content="苹果是红色的。",
                metadata={"source": "水果.txt", "page": 1, "relevance_score": 0.9},
            )
        ]
        self.stream_error = None
        self.fail_before_first_event = False
        self.closed_streams = 0

    def count(self):
        return self.chunk_count

    def add_file(self, name, content):
        self.uploads.append((name, content))
        self.chunk_count += 2
        return 2

    def ask(self, question, history=None):
        self.calls.append((question, deepcopy(history)))
        return "苹果是红色的。[1]" if self.documents else "你好！", self.documents

    def ask_stream(self, question, history=None):
        self.calls.append((question, deepcopy(history)))
        try:
            if self.fail_before_first_event:
                raise self.stream_error
            yield {
                "type": "sources",
                "sources": [
                    {"content": document.page_content, "metadata": document.metadata}
                    for document in self.documents
                ],
                "mode": "knowledge" if self.documents else "model",
            }
            yield {"type": "token", "content": "苹"}
            yield {"type": "token", "content": "果\n"}
            if self.stream_error is not None:
                raise self.stream_error
            yield {"type": "done", "answer": "苹果\n"}
        finally:
            self.closed_streams += 1


class FakeTranscriber:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, *, filename="recording.wav", mime_type="audio/wav"):
        self.calls.append((audio, filename, mime_type))
        return "请介绍一下 Python。"


def sse_events(text):
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        data = json.loads(fields["data"])
        if fields["event"] != data["type"]:
            raise AssertionError("SSE event name must match its JSON type")
        events.append(data)
    return events


class APITests(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.transcriber = FakeTranscriber()
        api.get_rag_service.cache_clear()
        api.get_transcriber.cache_clear()
        api.app.dependency_overrides[api.get_rag_service] = lambda: self.service
        api.app.dependency_overrides[api.get_transcriber] = lambda: self.transcriber
        self.client = TestClient(api.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)
        self.addCleanup(api.app.dependency_overrides.clear)
        self.addCleanup(api.get_rag_service.cache_clear)
        self.addCleanup(api.get_transcriber.cache_clear)

    def test_health_does_not_initialize_services_or_read_configuration(self):
        api.app.dependency_overrides.clear()
        with (
            patch("settings.load_rag_settings", side_effect=AssertionError("must not load")) as settings,
            patch("speech.create_transcriber", side_effect=AssertionError("must not initialize")) as speech,
        ):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        settings.assert_not_called()
        speech.assert_not_called()

    def test_knowledge_status_uses_shared_service(self):
        response = self.client.get("/knowledge/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"count": 3})

    def test_document_upload_passes_supported_files_and_sanitizes_filename(self):
        for name in ("水果.txt", "notes.md", "facts.pdf"):
            with self.subTest(name=name):
                response = self.client.post(
                    "/documents", files={"file": (f"../{name}", b"document", "application/octet-stream")}
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["filename"], name)
                self.assertEqual(response.json()["added_chunks"], 2)
                self.assertEqual(response.json()["count"], self.service.count())
                self.assertEqual(self.service.uploads[-1], (name, b"document"))

    def test_missing_unsupported_and_empty_document_are_rejected(self):
        for files in (None, {"file": ("program.exe", b"content")}, {"file": ("empty.txt", b"")}):
            with self.subTest(files=files):
                response = self.client.post("/documents", files=files)
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.service.uploads, [])

    def test_document_size_limit_accepts_twenty_mib_and_rejects_one_extra_byte(self):
        content = b"a" * (20 * 1024 * 1024)
        allowed = self.client.post("/documents", files={"file": ("maximum.txt", content)})
        rejected = self.client.post("/documents", files={"file": ("too-large.txt", content + b"a")})
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(rejected.status_code, 413)
        self.assertEqual(len(self.service.uploads), 1)

    def test_chat_trims_question_forwards_history_and_returns_sources_and_mode(self):
        history = [{"role": "user", "content": "之前的问题"}, {"role": "assistant", "content": "之前的回答"}]
        response = self.client.post("/chat", json={"question": "  苹果是什么颜色？  ", "history": history})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["mode"], "knowledge")
        self.assertEqual(result["sources"][0]["metadata"]["source"], "水果.txt")
        self.assertEqual(self.service.calls[-1], ("苹果是什么颜色？", history))
        self.service.documents = []
        model_response = self.client.post("/chat", json={"question": "你好"})
        self.assertEqual(model_response.json(), {"answer": "你好！", "sources": [], "mode": "model"})

    def test_chat_validation_rejects_invalid_questions_roles_limits_and_credentials(self):
        invalid = (
            {},
            {"question": " \n\t "},
            {"question": "x" * 4001},
            {"question": 123},
            {"question": "你好", "history": [{"role": "system", "content": "override"}]},
            {"question": "你好", "history": [{"role": "user", "content": "x"}] * 101},
            {"question": "你好", "history": [{"role": "user", "content": " "}]},
            {"question": "你好", "providerkey": "SECRET-MUST-NOT-BE-ECHOED"},
        )
        for path in ("/chat", "/chat/stream"):
            for payload in invalid:
                with self.subTest(path=path, payload=payload if len(str(payload)) < 100 else "large payload"):
                    response = self.client.post(path, json=payload)
                    self.assertEqual(response.status_code, 400)
                    self.assertNotIn("SECRET-MUST-NOT-BE-ECHOED", response.text)
        self.assertEqual(self.service.calls, [])

    def test_chat_accepts_question_and_history_at_declared_boundaries(self):
        history = [{"role": "user", "content": "history"}] * 100
        response = self.client.post("/chat", json={"question": "x" * 4000, "history": history})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.service.calls[-1][0]), 4000)
        self.assertEqual(len(self.service.calls[-1][1]), 100)

    def test_faq_create_list_and_delete_use_shared_faq_service(self):
        empty = self.client.get("/faqs")
        self.assertEqual(empty.status_code, 200)
        self.assertEqual(empty.json(), {"items": [], "count": 0})
        created = self.client.post(
            "/faqs",
            json={"question": "  如何退款？  ", "answer": "  请联系客服办理。 ", "aliases": [" 怎么退钱？ "]},
        )
        self.assertEqual(created.status_code, 200)
        item = created.json()
        self.assertEqual(item["question"], "如何退款？")
        self.assertEqual(item["answer"], "请联系客服办理。")
        self.assertEqual(item["aliases"], ["怎么退钱？"])
        self.assertEqual(self.service.faq.calls[-1], ("add", "如何退款？", "请联系客服办理。", ["怎么退钱？"]))
        listed = self.client.get("/faqs")
        self.assertEqual(listed.json(), {"items": [item], "count": 1})
        deleted = self.client.delete(f"/faqs/{item['id']}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json(), {"deleted": True})
        self.assertEqual(self.client.delete(f"/faqs/{item['id']}").status_code, 404)
        self.assertEqual(self.client.get("/faqs").json(), {"items": [], "count": 0})

    def test_faq_create_rejects_invalid_fields_and_accepts_boundaries(self):
        secret = "SECRET-IN-UNKNOWN-FAQ-FIELD"
        invalid = (
            {},
            {"question": " ", "answer": "回答"},
            {"question": "问题", "answer": "\n"},
            {"question": "问题", "answer": 12},
            {"question": "q" * 4001, "answer": "回答"},
            {"question": "问题", "answer": "a" * 12001},
            {"question": "问题", "answer": "回答", "aliases": ["a"] * 21},
            {"question": "问题", "answer": "回答", "aliases": [""]},
            {"question": "问题", "answer": "回答", "aliases": ["a" * 4001]},
            {"question": "问题", "answer": "回答", "provider_key": secret},
        )
        for payload in invalid:
            with self.subTest(payload=tuple(payload)):
                response = self.client.post("/faqs", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertNotIn(secret, response.text)
        self.assertEqual(self.service.faq.calls, [])
        response = self.client.post(
            "/faqs", json={"question": "q" * 4000, "answer": "a" * 12000, "aliases": ["x" * 4000] * 20}
        )
        self.assertEqual(response.status_code, 200)

    def test_faq_import_passes_json_bytes_and_rejects_bad_uploads(self):
        content = '[{"question":"如何退款？","answer":"请联系客服。"}]'.encode("utf-8")
        response = self.client.post(
            "/faqs/import", files={"file": ("../customer-faq.JSON", content, "application/json")}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"added": 2, "skipped": 1, "count": 2})
        self.assertEqual(self.service.faq.calls, [("import", content)])
        for files in (None, {"file": ("faq.csv", content)}, {"file": ("empty.json", b"")}):
            with self.subTest(files="missing" if files is None else tuple(files)):
                self.assertEqual(self.client.post("/faqs/import", files=files).status_code, 400)
        self.assertEqual(len(self.service.faq.calls), 1)

    def test_faq_import_limits_upload_before_calling_service(self):
        content = b"a" * (1024 * 1024)
        allowed = self.client.post("/faqs/import", files={"file": ("maximum.json", content)})
        rejected = self.client.post("/faqs/import", files={"file": ("too-large.json", content + b"a")})
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(rejected.status_code, 413)
        self.assertEqual(len(self.service.faq.calls), 1)

    def test_faq_service_errors_do_not_expose_private_data(self):
        secret = "SECRET-IN-FAQ-ERROR"
        cases = (
            ("list", "/faqs", "get", None),
            ("add", "/faqs", "post", {"json": {"question": "退款", "answer": "联系客服"}}),
            ("delete", "/faqs/unknown", "delete", None),
            ("import_json", "/faqs/import", "post", {"files": {"file": ("faq.json", b"[]")}}),
        )
        for method, path, verb, arguments in cases:
            with self.subTest(method=method):
                with patch.object(self.service.faq, method, side_effect=ValueError(secret)):
                    response = getattr(self.client, verb)(path, **(arguments or {}))
                self.assertEqual(response.status_code, 400)
                self.assertNotIn(secret, response.text)

    def test_faq_mode_is_derived_from_source_metadata_for_normal_and_streaming_chat(self):
        self.service.documents = [Document(
            page_content="联系客服办理退款。",
            metadata={"kind": "faq", "faq_id": "faq-1", "source": "客服 FAQ", "relevance_score": 1.0},
        )]
        response = self.client.post("/chat", json={"question": "如何退款？"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "faq")
        events = sse_events(self.client.post("/chat/stream", json={"question": "如何退款？"}).text)
        self.assertEqual(events[0]["mode"], "faq")
        self.assertEqual(events[0]["sources"][0]["metadata"]["faq_id"], "faq-1")
        forged = api._stream_event({"type": "sources", "sources": [], "mode": "faq"})
        self.assertEqual(forged["mode"], "model")

    def test_faq_settings_get_and_update_share_service_and_preserve_optional_margin(self):
        response = self.client.get("/faqs/settings")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"threshold": 0.7, "min_margin": 0.05})
        response = self.client.put("/faqs/settings", json={"threshold": 0.75})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"threshold": 0.75, "min_margin": 0.05})
        self.assertEqual(self.service.faq.calls[-1], ("settings", {"threshold": 0.75}))
        response = self.client.put("/faqs/settings", json={"threshold": 0, "min_margin": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"threshold": 0.0, "min_margin": 1.0})
        self.assertEqual(self.client.get("/faqs/settings").json(), response.json())
        self.assertEqual(self.service.calls, [])

    def test_faq_settings_reject_invalid_ranges_non_numeric_values_and_extra_fields(self):
        payloads = (
            {},
            {"threshold": -0.01},
            {"threshold": 1.01},
            {"threshold": "0.70"},
            {"threshold": True},
            {"threshold": None},
            {"threshold": 0.7, "min_margin": -0.01},
            {"threshold": 0.7, "min_margin": 1.01},
            {"threshold": 0.7, "min_margin": "0.05"},
            {"threshold": 0.7, "provider_key": "SECRET-IN-FAQ-SETTINGS"},
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                response = self.client.put("/faqs/settings", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertNotIn("SECRET-IN-FAQ-SETTINGS", response.text)
        for field in ("threshold", "min_margin"):
            for value in (float("nan"), float("inf"), float("-inf")):
                payload = {"threshold": 0.7, field: value}
                response = self.client.put(
                    "/faqs/settings", content=json.dumps(payload), headers={"content-type": "application/json"}
                )
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.service.faq.calls, [])

    def test_faq_settings_failures_are_sanitized(self):
        secret = "SECRET-IN-FAQ-SETTINGS-ERROR"
        for method, verb, arguments in (
            ("get_settings", "get", {}),
            ("update_settings", "put", {"json": {"threshold": 0.7}}),
        ):
            with self.subTest(method=method):
                with patch.object(self.service.faq, method, side_effect=ValueError(secret)):
                    response = getattr(self.client, verb)("/faqs/settings", **arguments)
                self.assertEqual(response.status_code, 400)
                self.assertNotIn(secret, response.text)

    def test_faq_match_returns_preview_candidates_without_invoking_chat(self):
        response = self.client.post("/faqs/match", json={"question": "  怎样申请退款？  "})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["question"], "怎样申请退款？")
        self.assertTrue(result["matched"])
        self.assertEqual(result["reason"], "hybrid")
        self.assertEqual(result["match"]["answer"], "联系客服办理。")
        self.assertEqual(result["candidates"][0]["semantic_score"], 0.82)
        self.assertEqual(result["candidates"][0]["bm25_score"], 2.5)
        self.assertEqual(self.service.faq.calls, [("preview", "怎样申请退款？")])
        self.assertEqual(self.service.calls, [])
        self.service.faq.preview_result.update(matched=False, reason="below_threshold", match=None)
        rejected = self.client.post("/faqs/match", json={"question": "我可以退货吗？"})
        self.assertEqual(rejected.status_code, 200)
        self.assertFalse(rejected.json()["matched"])
        self.assertIsNone(rejected.json()["match"])
        self.assertEqual(rejected.json()["reason"], "below_threshold")

    def test_faq_match_validates_question_and_rejects_unrelated_fields(self):
        for payload in (
            {}, {"question": "\n"}, {"question": 123}, {"question": "q" * 4001},
            {"question": "问题", "history": []},
            {"question": "问题", "threshold": 0.5},
            {"question": "问题", "provider_key": "SECRET-IN-MATCH-REQUEST"},
        ):
            with self.subTest(fields=tuple(payload)):
                response = self.client.post("/faqs/match", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertNotIn("SECRET-IN-MATCH-REQUEST", response.text)
        self.assertEqual(self.service.faq.calls, [])
        self.assertEqual(self.client.post("/faqs/match", json={"question": "q" * 4000}).status_code, 200)

    def test_faq_match_returns_negative_cosine_candidates_as_valid_non_matches(self):
        self.service.faq.preview_result.update(matched=False, reason="below_threshold", match=None)
        self.service.faq.preview_result["candidates"][0].update(
            semantic_score=-0.6, bm25_score=0.0, hybrid_score=-0.48
        )
        response = self.client.post("/faqs/match", json={"question": "完全无关的问题"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertFalse(result["matched"])
        self.assertEqual(result["candidates"][0]["semantic_score"], -0.6)
        self.assertEqual(result["candidates"][0]["hybrid_score"], -0.48)
        self.assertEqual(result["threshold"], 0.7)

    def test_faq_match_errors_and_non_finite_scores_are_sanitized(self):
        secret = "SECRET-IN-MATCH-ERROR"
        with patch.object(self.service.faq, "preview", side_effect=ValueError(secret)):
            response = self.client.post("/faqs/match", json={"question": "退款"})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn(secret, response.text)
        for field, value in (
            ("semantic_score", float("nan")), ("semantic_score", 1.2),
            ("bm25_score", float("inf")), ("bm25_score", -1),
            ("hybrid_score", float("-inf")), ("hybrid_score", -1.1),
        ):
            with self.subTest(field=field, value=str(value)):
                result = deepcopy(self.service.faq.preview_result)
                result["candidates"][0][field] = value
                with patch.object(self.service.faq, "preview", return_value=result):
                    response = self.client.post("/faqs/match", json={"question": "退款"})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json(), {"detail": "请求内容无效，请检查输入、文件格式或文档内容。"})

    def test_service_errors_have_safe_status_codes_without_exception_text(self):
        secret = "SECRET-IN-PROVIDER-ERROR-AND-HEADERS"
        request = httpx.Request("POST", "https://example.invalid/chat", headers={"Authorization": secret})
        errors = [
            (APIConnectionError(message=secret, request=request), 503),
            (APITimeoutError(request=request), 503),
            (APIStatusError(secret, response=httpx.Response(401, request=request), body={"secret": secret}), 502),
            (APIStatusError(secret, response=httpx.Response(429, request=request), body=None), 503),
            (APIStatusError(secret, response=httpx.Response(500, request=request), body=None), 503),
            (ValueError(secret), 400),
            (RuntimeError(secret), 500),
        ]
        for error, expected_status in errors:
            with self.subTest(error=type(error).__name__, status=expected_status):
                with patch.object(self.service, "ask", side_effect=error), self.assertLogs("api", level="WARNING") as logs:
                    response = self.client.post("/chat", json={"question": "你好"})
                self.assertEqual(response.status_code, expected_status)
                self.assertNotIn(secret, response.text)
                self.assertNotIn(secret, " ".join(logs.output))

    def test_non_json_metadata_is_rejected_without_echoing_contents(self):
        for value in (object(), float("nan"), float("inf")):
            with self.subTest(value=type(value).__name__):
                self.service.documents = [Document(page_content="text", metadata={"invalid": value})]
                response = self.client.post("/chat", json={"question": "你好"})
                self.assertEqual(response.status_code, 500)
                self.assertNotIn("object at", response.text)

    def test_stream_has_named_json_events_and_forwards_history(self):
        history = [{"role": "user", "content": "我们在聊苹果。"}]
        response = self.client.post("/chat/stream", json={"question": "它是什么颜色？", "history": history})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = sse_events(response.text)
        self.assertEqual([event["type"] for event in events], ["sources", "token", "token", "done"])
        self.assertEqual(events[0]["mode"], "knowledge")
        self.assertEqual(events[0]["sources"][0]["metadata"]["source"], "水果.txt")
        self.assertEqual("".join(event["content"] for event in events if event["type"] == "token"), events[-1]["answer"])
        self.assertEqual(self.service.calls[-1], ("它是什么颜色？", history))
        self.assertEqual(self.service.closed_streams, 1)

    def test_stream_failure_is_sanitized_and_never_emits_done(self):
        secret = "SECRET-IN-STREAM-EXCEPTION"
        self.service.stream_error = APIConnectionError(
            message=secret, request=httpx.Request("POST", "https://example.invalid/chat")
        )
        for before_first_event in (False, True):
            with self.subTest(before_first_event=before_first_event):
                self.service.fail_before_first_event = before_first_event
                response = self.client.post("/chat/stream", json={"question": "你好"})
                self.assertEqual(response.status_code, 200)
                events = sse_events(response.text)
                self.assertEqual(events[-1]["type"], "error")
                self.assertEqual(events[-1]["status_code"], 503)
                self.assertNotIn("done", [event["type"] for event in events])
                self.assertNotIn(secret, response.text)
        self.assertEqual(self.service.closed_streams, 2)

    def test_disconnected_stream_closes_underlying_model_generator(self):
        response = api.chat_stream(api.ChatRequest(question="你好"), service=self.service)

        async def disconnect():
            async def receive():
                raise AssertionError("HTTP 2.4 stream does not need receive")

            async def send(message):
                if message["type"] == "http.response.body" and message.get("body"):
                    raise OSError("simulated disconnect")

            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

        with self.assertRaises(ClientDisconnect):
            asyncio.run(disconnect())
        self.assertEqual(self.service.closed_streams, 1)

    def test_audio_transcription_passes_audio_filename_and_mime(self):
        response = self.client.post(
            "/audio/transcriptions", files={"file": ("recording.wav", b"fake-wav", "audio/wav")}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "请介绍一下 Python。"})
        self.assertEqual(self.transcriber.calls, [(b"fake-wav", "recording.wav", "audio/wav")])
        empty = self.client.post("/audio/transcriptions", files={"file": ("empty.wav", b"", "audio/wav")})
        self.assertEqual(empty.status_code, 400)

    def test_audio_provider_errors_are_unavailable_and_sanitized(self):
        from speech import SpeechServiceError

        secret = "SECRET-IN-SPEECH-ERROR"
        with (
            patch.object(self.transcriber, "transcribe", side_effect=SpeechServiceError(secret)),
            self.assertLogs("api", level="WARNING") as logs,
        ):
            response = self.client.post(
                "/audio/transcriptions", files={"file": ("recording.wav", b"audio", "audio/wav")}
            )
        self.assertEqual(response.status_code, 502)
        self.assertNotIn(secret, response.text)
        self.assertNotIn(secret, " ".join(logs.output))

    def test_dependency_initialization_errors_are_sanitized(self):
        secret = "SECRET-IN-INITIALIZATION-ERROR"
        api.app.dependency_overrides.pop(api.get_rag_service)
        with patch("settings.load_rag_settings", side_effect=ValueError(secret)):
            response = self.client.post("/chat", json={"question": "你好"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(secret, response.text)
        api.app.dependency_overrides.pop(api.get_transcriber)
        with patch("speech.create_transcriber", side_effect=RuntimeError(secret)):
            response = self.client.post("/audio/transcriptions", files={"file": ("test.wav", b"audio", "audio/wav")})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(secret, response.text)


if __name__ == "__main__":
    unittest.main()
