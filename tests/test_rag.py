"""Offline behavior checks for the minimal RAG application.

Run with: python -m unittest discover -s tests -v
The tests use a real temporary Chroma database, but never call a model API.
"""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from pydantic import Field

from rag import RAGService, RagSettings, load_document


class LocalEmbeddings(Embeddings):
    """Small deterministic vectors with inspectable document embedding calls."""

    terms = ("苹果", "香蕉", "星系", "量子", "火山", "海洋")

    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []
        self.query_texts: list[str] = []

    def vector(self, text: str) -> list[float]:
        return [float(text.count(term)) for term in self.terms] + [0.01]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_texts.append(text)
        return self.vector(text)


class ChunkedChatModel(FakeListChatModel):
    """Emit inspectable real model chunks; a non-streaming call is a test failure."""

    stream_chunks: list[str]
    fail_after: int | None = None
    produced_chunks: list[str] = Field(default_factory=list)
    captured_messages: list[list[BaseMessage]] = Field(default_factory=list)
    stream_closed: bool = False

    def _call(self, *args, **kwargs):
        raise AssertionError("Streaming must not call invoke or generate first")

    def _stream(self, messages, **kwargs):
        self.captured_messages.append(messages)
        try:
            for index, content in enumerate(self.stream_chunks):
                if self.fail_after == index:
                    raise RuntimeError("simulated model stream interruption")
                self.produced_chunks.append(content)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=content,
                        chunk_position="last" if index == len(self.stream_chunks) - 1 else None,
                    )
                )
        finally:
            self.stream_closed = True


class RAGTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="minimal-rag-test-", ignore_cleanup_errors=True
        )
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_directory = Path(self.temporary_directory.name) / "chroma"
        self.embeddings = LocalEmbeddings()

    def settings(self, **overrides: object) -> RagSettings:
        values = {
            "api_key": "test-key-with-no-network-access",
            "base_url": "https://example.invalid/compatible-mode/v1",
            "persist_directory": self.database_directory,
        }
        values.update(overrides)
        return RagSettings(**values)

    def service(self, **overrides: object) -> RAGService:
        return RAGService(
            self.settings(**overrides),
            embeddings=self.embeddings,
            llm=FakeListChatModel(responses=["苹果是红色的。[1]"]),
        )

    def test_text_encodings_preserve_chinese(self) -> None:
        text = "苹果是红色的。\n这是中文知识库。"
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            with self.subTest(encoding=encoding):
                documents = load_document("知识.txt", text.encode(encoding))
                self.assertEqual(len(documents), 1)
                self.assertEqual(documents[0].page_content.strip(), text)
                self.assertEqual(documents[0].metadata["source"], "知识.txt")

    def test_markdown_preserves_original_text(self) -> None:
        text = "# 水果\n\n苹果是红色的。\n"
        documents = load_document("水果.md", text.encode("utf-8"))
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content.strip(), text.strip())
        self.assertEqual(documents[0].metadata["source"], "水果.md")

    def test_invalid_files_are_rejected(self) -> None:
        for name, content in (
            ("empty.txt", b""),
            ("whitespace.md", b" \t\r\n "),
            ("unsupported.exe", b"arbitrary content"),
            ("broken.pdf", b"this is not a PDF file"),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    load_document(name, content)

    def test_pdf_without_extractable_text_is_rejected(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        output = BytesIO()
        writer.write(output)
        with self.assertRaises(ValueError):
            load_document("blank.pdf", output.getvalue())

    def test_pdf_text_keeps_one_based_page_number(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        text_page = writer.add_blank_page(width=200, height=200)
        text_page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): DictionaryObject(
                            {
                                NameObject("/Type"): NameObject("/Font"),
                                NameObject("/Subtype"): NameObject("/Type1"),
                                NameObject("/BaseFont"): NameObject("/Helvetica"),
                            }
                        )
                    }
                )
            }
        )
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 20 100 Td (Apples are red.) Tj ET")
        text_page[NameObject("/Contents")] = writer._add_object(stream)
        output = BytesIO()
        writer.write(output)
        documents = load_document("facts.pdf", output.getvalue())
        self.assertEqual(len(documents), 1)
        self.assertIn("Apples are red.", documents[0].page_content)
        self.assertEqual(documents[0].metadata["source"], "facts.pdf")
        self.assertEqual(documents[0].metadata["page"], 2)

    def test_new_database_starts_empty(self) -> None:
        service = self.service()
        self.assertEqual(service.count(), 0)
        self.assertEqual(self.embeddings.document_batches, [])

    def test_ingestion_reports_and_stores_chunks(self) -> None:
        service = self.service(chunk_size=100, chunk_overlap=15)
        text = "\n".join(f"条目 {index}：苹果是红色的水果。" for index in range(50))
        added = service.add_file("水果.txt", text.encode("utf-8"))
        self.assertGreater(added, 1)
        self.assertEqual(service.count(), added)
        self.assertEqual(
            sum(len(batch) for batch in self.embeddings.document_batches), added
        )

    def test_duplicate_upload_skips_embedding_and_storage(self) -> None:
        service = self.service()
        content = "苹果是红色的水果。".encode("utf-8")
        added = service.add_file("水果.txt", content)
        previous_batches = [list(batch) for batch in self.embeddings.document_batches]
        self.assertGreater(added, 0)
        self.assertEqual(service.add_file("水果.txt", content), 0)
        self.assertEqual(service.count(), added)
        self.assertEqual(self.embeddings.document_batches, previous_batches)

    def test_failed_embedding_batch_retries_only_missing_chunks(self) -> None:
        class FailsOnceOnSecondBatch(LocalEmbeddings):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated embedding interruption")
                return super().embed_documents(texts)

        embeddings = FailsOnceOnSecondBatch()
        service = RAGService(
            self.settings(chunk_size=80, chunk_overlap=10),
            embeddings=embeddings,
            llm=FakeListChatModel(responses=["苹果是红色的。[1]"]),
        )
        text = "\n".join(
            f"条目 {index}：苹果是红色的水果，含有膳食纤维。"
            for index in range(60)
        )
        content = text.encode("utf-8")
        with self.assertRaisesRegex(RuntimeError, "simulated embedding interruption"):
            service.add_file("水果.txt", content)
        stored_before_retry = service.count()
        self.assertEqual(stored_before_retry, 10)
        first_batch = list(embeddings.document_batches[0])
        added_on_retry = service.add_file("水果.txt", content)
        self.assertGreater(added_on_retry, 0)
        self.assertEqual(service.count(), stored_before_retry + added_on_retry)
        self.assertEqual(
            sum(len(batch) for batch in embeddings.document_batches), service.count()
        )
        self.assertTrue(
            all(
                text not in first_batch
                for batch in embeddings.document_batches[1:]
                for text in batch
            )
        )
        self.assertEqual(service.add_file("水果.txt", content), 0)

    def test_database_can_be_reloaded_without_reembedding(self) -> None:
        service = self.service()
        expected = service.add_file("持久化.txt", "苹果是红色的。".encode("utf-8"))
        previous_batches = [list(batch) for batch in self.embeddings.document_batches]
        reloaded = self.service()
        self.assertEqual(reloaded.count(), expected)
        self.assertEqual(self.embeddings.document_batches, previous_batches)
        answer, sources = reloaded.ask("苹果是什么颜色？")
        self.assertIn("红色", answer)
        self.assertEqual(sources[0].metadata["source"], "持久化.txt")

    def test_retrieval_returns_relevant_original_source(self) -> None:
        prompts = []

        def answer_from_prompt(prompt):
            prompts.append(prompt.to_string())
            return AIMessage(content="苹果是红色的。[1]")

        service = RAGService(
            self.settings(top_k=1),
            embeddings=self.embeddings,
            llm=RunnableLambda(answer_from_prompt),
        )
        apple_text = "苹果是红色的水果。苹果含有膳食纤维。"
        service.add_file("水果.txt", apple_text.encode("utf-8"))
        service.add_file("天文.txt", "星系由许多恒星组成。星系尺度巨大。".encode("utf-8"))
        answer, sources = service.ask("苹果是什么颜色？")
        self.assertIsInstance(answer, str)
        self.assertIn("红色", answer)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].metadata["source"], "水果.txt")
        self.assertEqual(sources[0].page_content, apple_text)
        self.assertGreaterEqual(sources[0].metadata["relevance_score"], 0.35)
        self.assertEqual(len(prompts), 1)
        self.assertIn(apple_text, prompts[0])
        self.assertIn("苹果是什么颜色？", prompts[0])
        self.assertIn("水果.txt", prompts[0])
        self.assertIn("[1]", prompts[0])
        self.assertIn("本轮本地参考资料", prompts[0])
        self.assertNotIn("星系由许多恒星组成", prompts[0])

    def test_blank_questions_are_rejected_before_processing(self) -> None:
        service = self.service()
        for question in ("", " \t\n "):
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    service.ask(question)
        self.assertEqual(self.embeddings.query_texts, [])

    def test_empty_database_chats_with_history_without_embedding_or_rewrite(self) -> None:
        prompts = []

        def respond(prompt):
            prompts.append(prompt.to_string())
            return AIMessage(content="你叫小林。")

        service = RAGService(
            self.settings(), embeddings=self.embeddings, llm=RunnableLambda(respond)
        )
        history = [
            {"role": "user", "content": "我叫小林。"},
            {"role": "assistant", "content": "你好，小林！"},
        ]
        with patch.object(
            self.embeddings, "embed_query", side_effect=AssertionError("must not embed")
        ) as embed_query:
            answer, sources = service.ask("我的名字是什么？", history=history)
        self.assertEqual(answer, "你叫小林。")
        self.assertEqual(sources, [])
        embed_query.assert_not_called()
        self.assertEqual(len(prompts), 1)
        self.assertIn("我叫小林。", prompts[0])
        self.assertIn("你好，小林！", prompts[0])
        self.assertIn("我的名字是什么？", prompts[0])

    def test_unrelated_documents_do_not_become_sources_or_context(self) -> None:
        prompts = []

        def respond(prompt):
            prompts.append(prompt.to_string())
            return AIMessage(content="量子纠缠是量子系统之间的一种关联。")

        service = RAGService(
            self.settings(), embeddings=self.embeddings, llm=RunnableLambda(respond)
        )
        service.add_file("水果.txt", "苹果是红色的。".encode("utf-8"))
        answer, sources = service.ask("量子纠缠是什么？")
        self.assertIn("量子纠缠", answer)
        self.assertEqual(sources, [])
        self.assertNotIn("水果.txt", prompts[0])
        self.assertNotIn("苹果是红色的。", prompts[0])
        self.assertEqual(self.embeddings.query_texts, ["量子纠缠是什么？"])

    def test_no_sources_routes_to_plain_chat_prompt_with_history(self) -> None:
        prompts = []
        question = "量子纠缠是什么？"
        responses = iter(["普通聊天回复。", question, "普通聊天回复。"])

        def respond(prompt):
            prompts.append(prompt)
            return AIMessage(content=next(responses))

        service = RAGService(
            self.settings(), embeddings=self.embeddings, llm=RunnableLambda(respond)
        )
        history = [
            {"role": "user", "content": "我正在学习物理。"},
            {"role": "assistant", "content": "我们可以聊聊物理概念。"},
        ]
        for with_documents in (False, True):
            with self.subTest(with_documents=with_documents):
                if with_documents:
                    service.add_file("水果.txt", "苹果是红色的。".encode("utf-8"))
                _, sources = service.ask(question, history=history)
                self.assertEqual(sources, [])
                messages = prompts[-1].to_messages()
                self.assertEqual(messages[-1].type, "human")
                self.assertEqual(messages[-1].content, question)
                prompt_text = prompts[-1].to_string()
                self.assertNotIn("本轮本地参考资料", prompt_text)
                self.assertNotIn("水果.txt", prompt_text)
                self.assertIn("我正在学习物理。", prompt_text)
                self.assertIn("我们可以聊聊物理概念。", prompt_text)
        self.assertEqual(len(prompts), 3)  # Empty chat, then rewrite and unrelated chat.

    def test_cosine_threshold_includes_boundary_and_filters_lower_scores(self) -> None:
        class FixedEmbeddings(LocalEmbeddings):
            vectors = {
                "门槛以下": [5.0, 9.0],
                "门槛边界": [6.0, 8.0],
                "最高相关": [1.0, 0.0],
                "查询": [2.0, 0.0],
            }

            def vector(self, text):
                return self.vectors[text]

        prompts = []

        def respond(prompt):
            prompts.append(prompt.to_string())
            return AIMessage(content="已参考相关内容。[1][2]")

        service = RAGService(
            self.settings(top_k=3, relevance_threshold=0.6),
            embeddings=FixedEmbeddings(),
            llm=RunnableLambda(respond),
        )
        for name, text in (
            ("低相关.txt", "门槛以下"),
            ("边界.txt", "门槛边界"),
            ("最高.txt", "最高相关"),
        ):
            service.add_file(name, text.encode("utf-8"))
        _, sources = service.ask("查询")
        self.assertEqual(
            [document.metadata["source"] for document in sources],
            ["最高.txt", "边界.txt"],
        )
        self.assertAlmostEqual(sources[0].metadata["relevance_score"], 1.0)
        self.assertAlmostEqual(sources[1].metadata["relevance_score"], 0.6)
        self.assertNotIn("门槛以下", prompts[0])

    def test_changing_threshold_reuses_existing_index(self) -> None:
        service = self.service(relevance_threshold=0.35)
        content = "苹果是红色的。".encode("utf-8")
        count = service.add_file("水果.txt", content)
        previous_batches = list(self.embeddings.document_batches)
        stricter_service = self.service(relevance_threshold=0.9)
        self.assertEqual(stricter_service.count(), count)
        self.assertEqual(stricter_service.add_file("水果.txt", content), 0)
        self.assertEqual(self.embeddings.document_batches, previous_batches)

    def test_history_rewrites_retrieval_and_reaches_final_answer_prompt(self) -> None:
        prompts = []

        def respond(prompt):
            prompts.append(prompt.to_string())
            if len(prompts) == 1:
                return AIMessage(content="苹果是什么颜色？")
            return AIMessage(content="苹果是红色的。[1]")

        service = RAGService(
            self.settings(top_k=1),
            embeddings=self.embeddings,
            llm=RunnableLambda(respond),
        )
        service.add_file("水果.txt", "苹果是红色的。".encode("utf-8"))
        service.add_file("天文.txt", "星系由恒星组成。".encode("utf-8"))
        history = [
            {"role": "user", "content": "我们来聊苹果。"},
            {"role": "assistant", "content": "你想了解苹果的什么？"},
        ]
        answer, sources = service.ask("那它是什么颜色？", history=history)
        self.assertEqual(self.embeddings.query_texts, ["苹果是什么颜色？"])
        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertIn("我们来聊苹果。", prompt)
            self.assertIn("你想了解苹果的什么？", prompt)
            self.assertIn("那它是什么颜色？", prompt)
        self.assertIn("苹果是红色的。", prompts[-1])
        self.assertIn("本轮本地参考资料", prompts[-1])
        self.assertIn("红色", answer)
        self.assertEqual(sources[0].metadata["source"], "水果.txt")

    def test_retrieval_failures_fall_back_and_log_only_error_type(self) -> None:
        service = RAGService(
            self.settings(),
            embeddings=self.embeddings,
            llm=FakeListChatModel(responses=["我可以继续与你聊天。"]),
        )
        service.add_file("水果.txt", "苹果是红色的。".encode("utf-8"))
        collection_type = type(service.client.get_collection(service.collection_name))
        secret = "SECRET-MUST-NOT-APPEAR-IN-RETRIEVAL-LOGS"
        for stage in ("embedding", "database-query"):
            failing_call = (
                patch.object(self.embeddings, "embed_query", side_effect=RuntimeError(secret))
                if stage == "embedding"
                else patch.object(collection_type, "query", side_effect=RuntimeError(secret))
            )
            with self.subTest(stage=stage):
                with failing_call as called, self.assertLogs("rag", level="WARNING") as logs:
                    answer, sources = service.ask("苹果是什么颜色？")
                called.assert_called_once()
                self.assertEqual(answer, "我可以继续与你聊天。")
                self.assertEqual(sources, [])
                self.assertIn("RuntimeError", " ".join(logs.output))
                self.assertNotIn(secret, " ".join(logs.output))

    def test_invalid_query_vectors_cannot_produce_sources(self) -> None:
        service = RAGService(
            self.settings(),
            embeddings=self.embeddings,
            llm=FakeListChatModel(responses=["我可以正常回答。"]),
        )
        service.add_file("水果.txt", "苹果是红色的。".encode("utf-8"))
        for vector in ([0.0] * 7, [float("nan")] * 7, [float("inf")] * 7):
            with self.subTest(vector=vector):
                with patch.object(self.embeddings, "embed_query", return_value=vector):
                    answer, sources = service.ask("苹果是什么颜色？")
                self.assertEqual(answer, "我可以正常回答。")
                self.assertEqual(sources, [])

    def test_final_chat_errors_are_propagated_with_or_without_documents(self) -> None:
        error = RuntimeError("simulated final chat failure")

        def fail(_prompt):
            raise error

        service = RAGService(
            self.settings(), embeddings=self.embeddings, llm=RunnableLambda(fail)
        )
        for with_documents in (False, True):
            with self.subTest(with_documents=with_documents):
                if with_documents:
                    service.add_file("水果.txt", "苹果是红色的。".encode("utf-8"))
                with self.assertRaises(RuntimeError) as raised:
                    service.ask("苹果是什么颜色？")
                self.assertIs(raised.exception, error)

    def test_stream_delivers_each_model_chunk_before_consuming_the_next(self) -> None:
        model = ChunkedChatModel(responses=["unused"], stream_chunks=["你", "好", "！"])
        service = RAGService(self.settings(), embeddings=self.embeddings, llm=model)
        stream = service.ask_stream("你好")
        self.addCleanup(stream.close)
        self.assertEqual(model.produced_chunks, [])
        self.assertEqual(
            next(stream), {"type": "sources", "sources": [], "mode": "model"}
        )
        self.assertEqual(model.produced_chunks, [])
        self.assertEqual(next(stream), {"type": "token", "content": "你"})
        self.assertEqual(model.produced_chunks, ["你"])
        self.assertEqual(next(stream), {"type": "token", "content": "好"})
        self.assertEqual(model.produced_chunks, ["你", "好"])
        self.assertEqual(next(stream), {"type": "token", "content": "！"})
        self.assertEqual(next(stream), {"type": "done", "answer": "你好！"})
        self.assertTrue(model.stream_closed)
        self.assertEqual(self.embeddings.query_texts, [])
        with self.assertRaises(StopIteration):
            next(stream)

    def test_stream_knowledge_sources_and_history_use_only_one_retrieval(self) -> None:
        model = ChunkedChatModel(
            responses=["unused"], stream_chunks=["苹果", "是红色的。", "[1]"]
        )
        service = RAGService(
            self.settings(top_k=1), embeddings=self.embeddings, llm=model
        )
        apple_text = "苹果是红色的水果。"
        service.add_file("水果.txt", apple_text.encode("utf-8"))
        service.add_file("天文.txt", "星系由许多恒星组成。".encode("utf-8"))
        rewrite = Mock()
        rewrite.invoke.return_value = "苹果是什么颜色？"
        service.query_rewrite_chain = rewrite
        history = [
            {"role": "user", "content": "我们来聊苹果。"},
            {"role": "assistant", "content": "你想了解苹果的什么？"},
        ]
        with patch.object(service, "_retrieve", wraps=service._retrieve) as retrieve:
            events = list(service.ask_stream("那它是什么颜色？", history=history))
        retrieve.assert_called_once()
        rewrite.invoke.assert_called_once()
        self.assertEqual(self.embeddings.query_texts, ["苹果是什么颜色？"])
        self.assertEqual(events[0]["type"], "sources")
        self.assertEqual(events[0]["mode"], "knowledge")
        self.assertEqual(len(events[0]["sources"]), 1)
        self.assertEqual(events[0]["sources"][0]["content"], apple_text)
        self.assertEqual(events[0]["sources"][0]["metadata"]["source"], "水果.txt")
        self.assertGreaterEqual(
            events[0]["sources"][0]["metadata"]["relevance_score"], 0.35
        )
        self.assertEqual(
            [event["content"] for event in events if event["type"] == "token"],
            ["苹果", "是红色的。", "[1]"],
        )
        self.assertEqual(events[-1], {"type": "done", "answer": "苹果是红色的。[1]"})
        self.assertEqual(len(model.captured_messages), 1)
        final_messages = model.captured_messages[0]
        self.assertEqual(final_messages[1].content, history[0]["content"])
        self.assertEqual(final_messages[2].content, history[1]["content"])
        self.assertIn("那它是什么颜色？", final_messages[-1].content)
        self.assertIn(apple_text, final_messages[-1].content)
        self.assertNotIn("星系由许多恒星组成", final_messages[-1].content)

    def test_stream_unrelated_documents_use_model_mode_with_empty_sources(self) -> None:
        model = ChunkedChatModel(responses=["unused"], stream_chunks=["量子系统有关联。"])
        service = RAGService(self.settings(), embeddings=self.embeddings, llm=model)
        service.add_file("水果.txt", "苹果是红色的。".encode("utf-8"))
        events = list(service.ask_stream("量子纠缠是什么？"))
        self.assertEqual(events[0], {"type": "sources", "sources": [], "mode": "model"})
        self.assertEqual(model.captured_messages[0][-1].content, "量子纠缠是什么？")
        self.assertEqual(self.embeddings.query_texts, ["量子纠缠是什么？"])

    def test_stream_failure_propagates_without_done_and_releases_model(self) -> None:
        model = ChunkedChatModel(
            responses=["unused"], stream_chunks=["第一段", "第二段"], fail_after=1
        )
        service = RAGService(self.settings(), embeddings=self.embeddings, llm=model)
        stream = service.ask_stream("你好")
        self.addCleanup(stream.close)
        events = [next(stream), next(stream)]
        with self.assertRaisesRegex(RuntimeError, "simulated model stream interruption"):
            next(stream)
        self.assertEqual([event["type"] for event in events], ["sources", "token"])
        self.assertTrue(model.stream_closed)
        with self.assertRaises(StopIteration):
            next(stream)

    def test_closing_answer_iterator_releases_model_without_draining_it(self) -> None:
        model = ChunkedChatModel(
            responses=["unused"], stream_chunks=["第一段", "尚未请求的第二段"]
        )
        service = RAGService(self.settings(), embeddings=self.embeddings, llm=model)
        stream = service.ask_stream("你好")
        next(stream)  # Sources do not start the model stream.
        self.assertEqual(next(stream), {"type": "token", "content": "第一段"})
        stream.close()
        self.assertTrue(model.stream_closed)
        self.assertEqual(model.produced_chunks, ["第一段"])

    def test_closing_answer_iterator_calls_underlying_stream_close(self) -> None:
        service = self.service()
        chunks = Mock()
        chunks.__iter__ = Mock(return_value=iter(["第一段", "第二段"]))
        model = Mock()
        model.stream.return_value = chunks
        service.llm = model
        stream = service.ask_stream("你好")
        next(stream)
        self.assertEqual(next(stream), {"type": "token", "content": "第一段"})
        stream.close()
        chunks.close.assert_called_once()
        model.invoke.assert_not_called()

    def test_empty_model_responses_raise_readable_errors_without_done(self) -> None:
        model = ChunkedChatModel(responses=["unused"], stream_chunks=["", " \n"])
        service = RAGService(self.settings(), embeddings=self.embeddings, llm=model)
        stream = service.ask_stream("你好")
        events = []
        with self.assertRaisesRegex(ValueError, "模型没有返回有效回答"):
            while True:
                events.append(next(stream))
        self.assertNotIn("done", [event["type"] for event in events])
        self.assertNotIn("", [event["content"] for event in events if event["type"] == "token"])
        self.assertTrue(model.stream_closed)
        service = RAGService(
            self.settings(),
            embeddings=self.embeddings,
            llm=FakeListChatModel(responses=[" \n"]),
        )
        with self.assertRaisesRegex(ValueError, "模型没有返回有效回答"):
            service.ask("你好")

    def test_exact_faq_answers_bypass_model_retrieval_and_embeddings(self) -> None:
        model = ChunkedChatModel(responses=["unused"], stream_chunks=["不应调用模型"])
        service = RAGService(self.settings(), embeddings=self.embeddings, llm=model)
        answer_text = "请在订单页申请售后，随后查看处理进度。"
        item = service.faq.add("苹果如何退货？", answer_text, ["我想退苹果"])
        history = [{"role": "user", "content": "我刚买了一箱水果。"}]
        with patch.object(service, "_retrieve") as retrieve:
            answer, sources = service.ask("苹果如何退货？", history=history)
            events = list(service.ask_stream("我想退苹果", history=history))
        retrieve.assert_not_called()
        self.assertEqual(answer, answer_text)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].metadata["kind"], "faq")
        self.assertEqual(sources[0].metadata["faq_id"], item["id"])
        self.assertEqual(events[0]["mode"], "faq")
        self.assertEqual(events[0]["sources"][0]["metadata"]["kind"], "faq")
        self.assertEqual(events[1:], [
            {"type": "token", "content": answer_text},
            {"type": "done", "answer": answer_text},
        ])
        self.assertEqual(model.produced_chunks, [])
        self.assertEqual(model.captured_messages, [])
        self.assertEqual(self.embeddings.document_batches, [])
        self.assertEqual(self.embeddings.query_texts, [])

    def test_faq_miss_preserves_model_and_document_answer_routes(self) -> None:
        model = ChunkedChatModel(responses=["unused"], stream_chunks=["模型继续回答。"])
        service = RAGService(self.settings(), embeddings=self.embeddings, llm=model)
        service.faq.add("香蕉如何退货？", "请申请香蕉售后。")
        events = list(service.ask_stream("量子是什么？"))
        self.assertEqual(events[0], {"type": "sources", "sources": [], "mode": "model"})
        service.add_file("水果.txt", "苹果是红色的。".encode())
        events = list(service.ask_stream("苹果是什么颜色？"))
        self.assertEqual(events[0]["mode"], "knowledge")
        self.assertEqual(events[0]["sources"][0]["metadata"]["source"], "水果.txt")
        self.assertEqual(len(model.captured_messages), 2)

    def test_faq_failure_falls_back_without_logging_exception_content(self) -> None:
        service = self.service()
        secret = "SECRET-FAQ-ERROR-MUST-NOT-BE-LOGGED"
        with (
            patch.object(service.faq, "match", side_effect=RuntimeError(secret)),
            self.assertLogs("rag", level="WARNING") as logs,
        ):
            answer, sources = service.ask("你好")
            events = list(service.ask_stream("你好"))
        self.assertEqual(answer, "苹果是红色的。[1]")
        self.assertEqual(sources, [])
        self.assertEqual(events[0]["mode"], "model")
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("RuntimeError", " ".join(logs.output))
        self.assertNotIn(secret, " ".join(logs.output))

    def test_faq_persistence_is_separate_from_document_collection(self) -> None:
        service = self.service()
        count = service.add_file("水果.txt", "苹果是红色的。".encode())
        collection = service.collection_name
        item = service.faq.add("香蕉如何退货？", "请申请售后。", ["香蕉退货"])
        self.assertEqual(service.count(), count)
        reloaded = self.service()
        self.assertEqual(reloaded.collection_name, collection)
        self.assertEqual(reloaded.count(), count)
        self.assertEqual(reloaded.faq.count(), 1)
        answer, sources = reloaded.ask("香蕉退货")
        self.assertEqual(answer, "请申请售后。")
        self.assertEqual(sources[0].metadata["faq_id"], item["id"])
        _, documents = reloaded.ask("苹果是什么颜色？")
        self.assertEqual(documents[0].metadata["source"], "水果.txt")
        self.assertNotIn("kind", documents[0].metadata)

    def test_embedding_model_change_uses_a_separate_collection(self) -> None:
        first = self.service(embedding_model="text-embedding-v4")
        original_count = first.add_file("水果.txt", "苹果是红色的。".encode("utf-8"))
        second = self.service(embedding_model="another-embedding-model")
        self.assertEqual(second.count(), 0)
        second.add_file("天文.txt", "星系由许多恒星组成。".encode("utf-8"))
        self.assertEqual(first.count(), original_count)
        reloaded_first = self.service(embedding_model="text-embedding-v4")
        self.assertEqual(reloaded_first.count(), original_count)
        _, sources = reloaded_first.ask("苹果是什么颜色？")
        self.assertEqual(sources[0].metadata["source"], "水果.txt")

    def test_provider_requests_are_openai_compatible_without_network(self) -> None:
        embedding_requests = []
        chat_requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.host, "example.invalid")
            body = json.loads(request.content)
            if request.url.path.endswith("/embeddings"):
                embedding_requests.append(body)
                texts = body["input"]
                self.assertIsInstance(texts, list)
                self.assertTrue(all(isinstance(text, str) for text in texts))
                return httpx.Response(
                    200,
                    json={
                        "object": "list",
                        "model": "text-embedding-v4",
                        "data": [
                            {
                                "object": "embedding",
                                "index": index,
                                "embedding": self.embeddings.vector(text),
                            }
                            for index, text in enumerate(texts)
                        ],
                        "usage": {"prompt_tokens": 10, "total_tokens": 10},
                    },
                )
            if request.url.path.endswith("/chat/completions"):
                chat_requests.append(body)
                return httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-offline-test",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "qwen-plus",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "苹果是红色的。[1]",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                )
            self.fail(f"Unexpected API endpoint: {request.url.path}")

        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
            with (
                patch(
                    "rag.OpenAIEmbeddings",
                    side_effect=lambda **kwargs: OpenAIEmbeddings(
                        http_client=http_client, **kwargs
                    ),
                ),
                patch(
                    "rag.ChatOpenAI",
                    side_effect=lambda **kwargs: ChatOpenAI(
                        http_client=http_client, **kwargs
                    ),
                ),
            ):
                service = RAGService(self.settings(chunk_size=80, chunk_overlap=10))
                text = "\n".join(
                    f"条目 {index}：苹果是红色的水果，含有膳食纤维。"
                    for index in range(60)
                )
                self.assertGreater(service.add_file("水果.txt", text.encode()), 10)
                answer, sources = service.ask("苹果是什么颜色？")

        self.assertIn("红色", answer)
        self.assertTrue(sources)
        self.assertGreater(len(embedding_requests), 1)
        self.assertTrue(any(len(body["input"]) == 10 for body in embedding_requests))
        for body in embedding_requests:
            self.assertLessEqual(len(body["input"]), 10)
            self.assertEqual(body["model"], "text-embedding-v4")
            self.assertEqual(body["encoding_format"], "float")
        self.assertEqual(len(chat_requests), 1)
        self.assertEqual(chat_requests[0]["model"], "qwen-plus")
        self.assertIs(chat_requests[0]["enable_thinking"], False)
        self.assertTrue(chat_requests[0]["messages"])
        self.assertEqual(chat_requests[0]["messages"][0]["role"], "system")
        self.assertIn("本地知识智能助手", chat_requests[0]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
