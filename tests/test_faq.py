"""Offline FAQ persistence, matching, isolation and lazy-index regression checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings

from faq import FAQService, MAX_JSON_BYTES, normalize_question


class TestEmbeddings(Embeddings):
    def __init__(self, values: dict[str, list[float]] | None = None) -> None:
        self.values = values or {}
        self.document_batches = []
        self.query_texts = []
        self.fail = False

    def vector(self, text: str) -> list[float]:
        if text in self.values:
            return self.values[text]
        if any(term in text for term in ("退款", "退钱", "退回")):
            return [1.0, 0.0, 0.0]
        if any(term in text for term in ("发货", "配送", "快递")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        if self.fail:
            raise RuntimeError("simulated embedding provider failure")
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_texts.append(text)
        if self.fail:
            raise RuntimeError("simulated embedding provider failure")
        return self.vector(text)


class FAQTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="faq-demo-test-", ignore_cleanup_errors=True)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.client = chromadb.PersistentClient(
            path=str(self.root / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.embeddings = TestEmbeddings()
        self.service = self.make_service()

    def make_service(self, **overrides) -> FAQService:
        values = {
            "database_path": self.root / "faq.sqlite3",
            "client": self.client,
            "embeddings": self.embeddings,
            "collection_name": "faq_demo_test_v1",
        }
        values.update(overrides)
        return FAQService(**values)

    def dump(self, items) -> bytes:
        return json.dumps(items, ensure_ascii=False).encode("utf-8")

    def add_refund(self, **kwargs) -> dict:
        return self.service.add("如何申请退款？", "在订单详情中点击申请退款。", **kwargs)

    def test_empty_constructor_and_lookup_do_not_request_embeddings(self):
        self.assertEqual(self.service.count(), 0)
        self.assertEqual(self.service.list(), [])
        self.assertIsNone(self.service.match("如何申请退款？"))
        self.assertEqual(self.embeddings.document_batches, [])
        self.assertEqual(self.embeddings.query_texts, [])
        self.assertEqual(self.client.list_collections(), [])

    def test_add_persists_without_embeddings_and_returns_fields(self):
        item = self.add_refund(aliases=["退钱怎么操作？"])
        self.assertEqual(set(item), {"id", "question", "answer", "aliases", "updated_at"})
        self.assertEqual(item["aliases"], ["退钱怎么操作？"])
        restored = self.make_service()
        self.assertEqual(restored.list(), [item])
        self.assertEqual(restored.count(), 1)
        self.assertEqual(self.embeddings.document_batches, [])

    def test_exact_standard_question_returns_answer_without_model_calls(self):
        item = self.add_refund()
        self.embeddings.fail = True
        document = self.service.match("  如何 申请退款?!  ")
        self.assertEqual(document.page_content, item["answer"])
        self.assertEqual(document.metadata["kind"], "faq")
        self.assertEqual(document.metadata["faq_id"], item["id"])
        self.assertEqual(document.metadata["faq_question"], item["question"])
        self.assertEqual(document.metadata["source"], "FAQ：如何申请退款？")
        self.assertEqual(document.metadata["match_method"], "exact")
        self.assertEqual(document.metadata["relevance_score"], 1.0)
        self.assertEqual(self.embeddings.document_batches, [])
        self.assertEqual(self.embeddings.query_texts, [])

    def test_alias_exact_is_case_and_width_insensitive(self):
        self.service.add("如何查询订单？", "在个人中心查询。", ["Order STATUS？"])
        document = self.service.match("ＯＲＤＥＲ　ｓｔａｔｕｓ!")
        self.assertEqual(document.metadata["match_method"], "exact")
        self.assertEqual(document.page_content, "在个人中心查询。")
        self.assertEqual(self.embeddings.query_texts, [])

    def test_normalization_preserves_negation_digits_and_decimal_separators(self):
        self.assertNotEqual(normalize_question("支持退款"), normalize_question("不支持退款"))
        self.assertNotEqual(normalize_question("第1天"), normalize_question("第2天"))
        self.assertNotEqual(normalize_question("2.5小时"), normalize_question("25小时"))
        self.assertNotEqual(normalize_question("-1"), normalize_question("1"))
        self.assertNotEqual(normalize_question("24-48小时"), normalize_question("2448小时"))

    def test_punctuation_only_query_is_not_an_embedding_request(self):
        self.add_refund()
        self.assertIsNone(self.service.match("！？"))
        self.assertEqual(self.embeddings.document_batches, [])
        self.assertEqual(self.embeddings.query_texts, [])

    def test_duplicate_add_is_idempotent_and_deduplicates_aliases(self):
        item = self.add_refund(aliases=["如何申请退款", "退钱怎么操作？", "退钱怎么操作！"])
        duplicate = self.service.add("如何申请退款!", item["answer"], ["退钱怎么操作"])
        self.assertEqual(duplicate, item)
        self.assertEqual(item["aliases"], ["退钱怎么操作？"])
        self.assertEqual(self.service.count(), 1)

    def test_changed_answer_or_aliases_requires_delete_first(self):
        self.add_refund()
        for answer, aliases in (("请联系客服。", []), ("在订单详情中点击申请退款。", ["怎样退钱"])):
            with self.subTest(answer=answer, aliases=aliases):
                with self.assertRaisesRegex(ValueError, "先删除"):
                    self.service.add("如何申请退款", answer, aliases)
        self.assertEqual(self.service.count(), 1)

    def test_alias_collisions_rejected_in_both_directions(self):
        self.add_refund(aliases=["退钱怎么操作"])
        for question, aliases in (("退钱怎么操作？", []), ("如何退货", ["如何申请退款"])):
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    self.service.add(question, "测试答案", aliases)
        self.assertEqual(self.service.count(), 1)

    def test_delete_persists_and_releases_question_and_alias_keys(self):
        item = self.add_refund(aliases=["退钱怎么操作"])
        self.assertTrue(self.service.delete(item["id"]))
        self.assertFalse(self.service.delete(item["id"]))
        self.assertEqual(self.make_service().count(), 0)
        replacement = self.service.add("退钱怎么操作", "请从订单申请。")
        self.assertNotEqual(replacement["id"], item["id"])

    def test_import_is_idempotent_with_duplicates_in_same_file(self):
        row = {"question": "如何退款", "answer": "请从订单详情申请。", "aliases": ["如何退钱"]}
        content = self.dump([row, row, {"question": "何时发货", "answer": "付款后两天。"}])
        self.assertEqual(self.service.import_json(content), {"added": 2, "skipped": 1, "count": 2})
        self.assertEqual(self.service.import_json(content), {"added": 0, "skipped": 3, "count": 2})
        self.assertEqual(self.embeddings.document_batches, [])

    def test_import_conflict_rolls_back_all_new_rows(self):
        self.add_refund()
        content = self.dump([
            {"question": "何时发货", "answer": "付款后两天。"},
            {"question": "如何申请退款", "answer": "冲突答案"},
        ])
        with self.assertRaises(ValueError):
            self.service.import_json(content)
        self.assertEqual(self.service.count(), 1)
        self.assertEqual(self.service.list()[0]["question"], "如何申请退款？")

    def test_import_validation_happens_before_any_write(self):
        valid = {"question": "如何退款", "answer": "请从订单申请"}
        for invalid in (None, {}, {"question": "发货", "answer": ""}, {**valid, "api_key": "fake-secret"}):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError) as error:
                    self.service.import_json(self.dump([valid, invalid]))
                self.assertNotIn("fake-secret", str(error.exception))
                self.assertEqual(self.service.count(), 0)

    def test_import_format_size_and_item_bounds(self):
        for content in (b"", b"not json", b"\xff", b"{}", b"[]" + b" " * MAX_JSON_BYTES):
            with self.subTest(length=len(content)):
                with self.assertRaises(ValueError):
                    self.service.import_json(content)
        with self.assertRaises(ValueError):
            self.service.import_json(self.dump([{"question": "a", "answer": "b"}] * 501))
        self.assertEqual(self.service.import_json(b"\xef\xbb\xbf[]"), {"added": 0, "skipped": 0, "count": 0})

    def test_invalid_record_bounds_and_controls(self):
        for question, answer, aliases in (
            ("", "answer", None), ("?！", "answer", None), ("a" * 4001, "b", None),
            ("q", "a" * 12001, None), ("q", "answer", "not a list"),
            ("q", "answer", ["a"] * 21), ("q", "answer", [42]),
            ("q", "answer", ["?！"]), ("q\x00", "answer", None),
        ):
            with self.subTest(length=len(str(question))):
                with self.assertRaises(ValueError):
                    self.service.add(question, answer, aliases)
        self.assertEqual(self.service.count(), 0)

    def test_semantic_match_lazily_indexes_and_returns_canonical_answer(self):
        item = self.add_refund()
        self.service.add("何时发货", "付款后两天。")
        document = self.service.match("钱什么时候退回")
        self.assertEqual(document.page_content, item["answer"])
        self.assertEqual(document.metadata["match_method"], "semantic")
        self.assertAlmostEqual(document.metadata["relevance_score"], 1.0)
        self.assertEqual(sum(map(len, self.embeddings.document_batches)), 2)
        self.service.match("退钱多久到账")
        self.assertEqual(sum(map(len, self.embeddings.document_batches)), 2)

    def test_unrelated_and_close_candidates_do_not_match(self):
        self.add_refund()
        self.assertIsNone(self.service.match("火星上有什么"))
        self.service.add("退钱怎么操作", "另一个不同场景的退款答案")
        self.assertIsNone(self.service.match("钱什么时候退回"))

    def test_alias_vectors_are_aggregated_before_runner_up_margin(self):
        aliases = [f"退钱怎么操作{i}" for i in range(20)]
        item = self.add_refund(aliases=aliases)
        self.service.add("何时发货", "付款后两天。")
        document = self.service.match("钱什么时候退回")
        self.assertEqual(document.metadata["faq_id"], item["id"])
        self.assertTrue(all(len(batch) <= 10 for batch in self.embeddings.document_batches))
        self.assertEqual(sum(map(len, self.embeddings.document_batches)), 22)

    def test_many_aliases_cannot_hide_a_competing_faq(self):
        self.add_refund(aliases=[f"退钱怎么操作{i}" for i in range(20)])
        self.service.add("退钱要带什么", "另一条退款答案")
        self.assertIsNone(self.service.match("钱什么时候退回"))

    def test_other_process_changes_are_seen_and_stale_vectors_removed(self):
        item = self.add_refund()
        self.service.match("钱什么时候退回")
        other = self.make_service()
        self.assertTrue(other.delete(item["id"]))
        other.add("何时发货", "付款后两天。")
        self.assertIsNone(self.service.match("钱什么时候退回"))
        collection = self.client.get_collection(self.service.collection_name)
        metadata = collection.get(include=["metadatas"])["metadatas"]
        self.assertEqual(len(metadata), 1)
        self.assertNotEqual(metadata[0]["faq_id"], item["id"])
        self.assertEqual(self.service.match("快递多久能发出").page_content, "付款后两天。")

    def test_new_embedding_collection_rebuilds_from_same_sqlite(self):
        item = self.add_refund()
        self.service.match("钱什么时候退回")
        new_embeddings = TestEmbeddings()
        changed = self.make_service(embeddings=new_embeddings, collection_name="faq_demo_test_v2")
        self.assertEqual(changed.count(), 1)
        self.assertEqual(new_embeddings.document_batches, [])
        self.assertEqual(changed.match("钱什么时候退回").metadata["faq_id"], item["id"])
        self.assertEqual(sum(map(len, new_embeddings.document_batches)), 1)

    def test_index_missing_content_hash_is_repaired(self):
        self.add_refund()
        self.service.match("钱什么时候退回")
        collection = self.client.get_collection(self.service.collection_name)
        row = collection.get(include=["metadatas"])
        collection.update(ids=row["ids"], metadatas=[{"content_hash": "broken"}])
        self.assertIsNotNone(self.service.match("钱什么时候退回"))
        self.assertEqual(sum(map(len, self.embeddings.document_batches)), 2)

    def test_partial_index_failure_is_repaired_without_losing_faq_records(self):
        self.add_refund()
        self.embeddings.fail = True
        with self.assertRaises(RuntimeError):
            self.service.match("钱什么时候退回")
        self.assertEqual(self.service.count(), 1)
        self.assertEqual(self.service.match("如何申请退款").metadata["match_method"], "exact")
        self.embeddings.fail = False
        self.assertIsNotNone(self.service.match("钱什么时候退回"))

    def test_invalid_zero_query_vector_is_not_evidence(self):
        self.add_refund()
        self.embeddings.values["未知问题"] = [0.0, 0.0, 0.0]
        self.assertIsNone(self.service.match("未知问题"))

    def test_threshold_and_margin_validated(self):
        for threshold, margin in ((-1, 0.05), (1.1, 0.05), (float("nan"), 0.05), (0.75, -1)):
            with self.subTest(threshold=threshold, margin=margin):
                with self.assertRaises(ValueError):
                    self.make_service(threshold=threshold, min_margin=margin)

    def test_saved_settings_override_constructor_defaults_across_services(self):
        self.assertEqual(self.service.get_settings(), {"threshold": 0.70, "min_margin": 0.08})
        self.assertEqual(
            self.service.update_settings(0.81, 0.12), {"threshold": 0.81, "min_margin": 0.12}
        )
        other = self.make_service(threshold=0.55, min_margin=0.01)
        self.assertEqual(other.get_settings(), {"threshold": 0.81, "min_margin": 0.12})
        self.assertEqual(other.update_settings(0.70), {"threshold": 0.70, "min_margin": 0.12})
        self.assertEqual(self.service.get_settings(), {"threshold": 0.70, "min_margin": 0.12})
        self.assertEqual(self.embeddings.document_batches, [])

    def test_invalid_saved_settings_cannot_change_valid_settings(self):
        self.service.update_settings(0.70, 0.08)
        for threshold, margin in ((70, None), (float("nan"), None), (True, None), (0.7, -0.1)):
            with self.subTest(threshold=threshold, margin=margin):
                with self.assertRaises(ValueError):
                    self.service.update_settings(threshold, margin)
        self.assertEqual(self.service.get_settings(), {"threshold": 0.70, "min_margin": 0.08})

    def test_preview_exact_empty_and_match_share_decision(self):
        self.assertEqual(self.service.preview("如何退款")["reason"], "empty")
        item = self.add_refund()
        preview = self.service.preview("如何申请退款")
        document = self.service.match("如何申请退款")
        self.assertTrue(preview["matched"])
        self.assertEqual(preview["reason"], "exact")
        self.assertEqual(preview["match"], {
            "id": item["id"], "question": item["question"], "answer": item["answer"],
            "match_method": "exact",
        })
        self.assertEqual(preview["candidates"][0]["semantic_score"], 1.0)
        self.assertEqual(document.page_content, preview["match"]["answer"])
        self.assertEqual(document.metadata["hybrid_score"], 1.0)
        self.assertEqual(self.embeddings.query_texts, [])

    @staticmethod
    def cosine_vector(score: float) -> list[float]:
        return [score, math.sqrt(1 - score * score), 0.0]

    def test_bm25_resolves_close_semantic_candidates_using_chinese_keywords(self):
        self.embeddings.values.update({
            "退款怎么办": self.cosine_vector(0.90),
            "快递几天送达": self.cosine_vector(0.88),
            "快递什么时候到": [1.0, 0.0, 0.0],
        })
        self.service.add("退款怎么办", "请从订单申请。")
        shipping = self.service.add("快递几天送达", "请查看物流预计送达时间。")
        preview = self.service.preview("快递什么时候到")
        self.assertTrue(preview["matched"])
        self.assertEqual(preview["match"]["id"], shipping["id"])
        self.assertGreater(preview["candidates"][0]["bm25_score"], 0)
        document = self.service.match("快递什么时候到")
        self.assertEqual(document.page_content, shipping["answer"])
        self.assertEqual(document.metadata["retrieval_method"], "hybrid")
        self.assertEqual(document.metadata["match_method"], "semantic")
        self.assertAlmostEqual(document.metadata["relevance_score"], 0.88)

    def test_keyword_winner_below_70_never_becomes_faq_or_forces_another_answer(self):
        self.embeddings.values.update({
            "退款怎么办": self.cosine_vector(0.75),
            "快递几天送达": self.cosine_vector(0.69),
            "快递什么时候到": [1.0, 0.0, 0.0],
        })
        self.service.add("退款怎么办", "请从订单申请。")
        shipping = self.service.add("快递几天送达", "请查看物流。")
        preview = self.service.preview("快递什么时候到")
        self.assertFalse(preview["matched"])
        self.assertEqual(preview["reason"], "below_threshold")
        self.assertEqual(preview["candidates"][0]["id"], shipping["id"])
        self.assertGreater(preview["candidates"][0]["hybrid_score"], 0.70)
        self.assertLess(preview["candidates"][0]["semantic_score"], 0.70)
        self.assertIsNone(self.service.match("快递什么时候到"))

    def test_70_boundary_and_next_query_settings_take_effect_without_reembedding(self):
        self.embeddings.values.update({
            "快递几天送达": self.cosine_vector(0.70),
            "快递什么时候到": [1.0, 0.0, 0.0],
        })
        self.service.add("快递几天送达", "请查看物流。")
        self.assertIsNotNone(self.service.match("快递什么时候到"))
        document_calls = sum(map(len, self.embeddings.document_batches))
        other = self.make_service()
        other.update_settings(0.71)
        self.assertIsNone(self.service.match("快递什么时候到"))
        other.update_settings(0.70)
        self.assertIsNotNone(self.service.match("快递什么时候到"))
        self.assertEqual(sum(map(len, self.embeddings.document_batches)), document_calls)

    def test_no_keyword_overlap_preserves_semantic_score_and_saved_margin(self):
        self.embeddings.values.update({
            "退款怎么办": self.cosine_vector(0.85),
            "快递几天送达": self.cosine_vector(0.80),
            "银河问题": [1.0, 0.0, 0.0],
        })
        self.service.add("退款怎么办", "请从订单申请。")
        self.service.add("快递几天送达", "请查看物流。")
        preview = self.service.preview("银河问题")
        self.assertFalse(preview["matched"])
        self.assertEqual(preview["reason"], "ambiguous")
        for candidate in preview["candidates"]:
            self.assertEqual(candidate["bm25_score"], 0.0)
            self.assertEqual(candidate["hybrid_score"], candidate["semantic_score"])
        self.service.update_settings(0.70, 0.04)
        self.assertTrue(self.service.preview("银河问题")["matched"])

    def test_keywords_are_indexed_for_aliases_and_negation_numbers_are_not_stopped(self):
        self.embeddings.values.update({
            "物流时效": self.cosine_vector(0.65),
            "快递配送要多久": self.cosine_vector(0.82),
            "快递什么时候到": [1.0, 0.0, 0.0],
        })
        item = self.service.add("物流时效", "请查看物流。", ["快递配送要多久"])
        preview = self.service.preview("快递什么时候到")
        self.assertTrue(preview["matched"])
        self.assertEqual(preview["match"]["id"], item["id"])
        self.assertGreater(preview["candidates"][0]["bm25_score"], 0)
        tokens = self.service._tokens("如何不支持退款？编号24-48，退款2.5元")
        self.assertNotIn("如何", tokens)
        self.assertTrue(any("不" in token for token in tokens))
        self.assertIn("24", tokens)
        self.assertIn("48", tokens)
        # jieba separates decimals, but both number components remain searchable.
        self.assertIn("2", tokens)
        self.assertIn("5", tokens)

    def test_keyword_candidate_outside_vector_top_candidates_is_scored(self):
        self.embeddings.values["快递运输多久"] = [1.0, 0.0, 0.0]
        for index in range(22):
            question = f"服务条目{index}"
            self.embeddings.values[question] = self.cosine_vector(0.85)
            self.service.add(question, "其他答案。")
        self.embeddings.values["快递发货时效"] = self.cosine_vector(0.75)
        item = self.service.add("快递发货时效", "请查看物流。")
        preview = self.service.preview("快递运输多久")
        self.assertTrue(preview["matched"])
        self.assertEqual(preview["match"]["id"], item["id"])
        self.assertAlmostEqual(preview["candidates"][0]["semantic_score"], 0.75)

    def test_keyword_index_refreshes_after_other_service_adds_or_deletes(self):
        self.add_refund()
        self.service.preview("快递什么时候到")
        other = self.make_service()
        item = other.add("快递何时发货", "付款后两天。")
        preview = self.service.preview("快递什么时候到")
        self.assertTrue(any(candidate["id"] == item["id"] for candidate in preview["candidates"]))
        other.delete(item["id"])
        preview = self.service.preview("快递什么时候到")
        self.assertFalse(any(candidate["id"] == item["id"] for candidate in preview["candidates"]))

    def test_negative_cosine_remains_visible_and_cannot_match(self):
        self.embeddings.values.update({"快递几天送达": [-1.0, 0.0, 0.0], "银河问题": [1.0, 0.0, 0.0]})
        self.service.add("快递几天送达", "请查看物流。")
        preview = self.service.preview("银河问题")
        self.assertFalse(preview["matched"])
        self.assertEqual(preview["reason"], "below_threshold")
        self.assertEqual(preview["candidates"][0]["semantic_score"], -1.0)
        self.assertEqual(preview["candidates"][0]["hybrid_score"], -1.0)
        self.assertIsNone(self.service.match("银河问题"))

    def test_add_aliases_keeps_identity_answer_existing_aliases_and_exact_needs_no_embedding(self):
        item = self.add_refund(aliases=["用户自己添加的退款问法"])
        self.embeddings.fail = True
        updated = self.service.add_aliases(item["id"], ["退款在哪里", "退款入口在哪里"])
        self.assertEqual(updated["id"], item["id"])
        self.assertEqual(updated["question"], item["question"])
        self.assertEqual(updated["answer"], item["answer"])
        self.assertEqual(updated["aliases"], ["用户自己添加的退款问法", "退款在哪里", "退款入口在哪里"])
        self.assertNotEqual(updated["updated_at"], item["updated_at"])
        self.assertEqual(self.make_service().list(), [updated])
        document = self.service.match("退款在哪里？")
        self.assertEqual(document.metadata["faq_id"], item["id"])
        self.assertEqual(document.metadata["match_method"], "exact")
        self.assertEqual(document.page_content, item["answer"])
        self.assertEqual(self.embeddings.document_batches, [])
        self.assertEqual(self.embeddings.query_texts, [])

    def test_add_aliases_is_idempotent_and_deduplicates_normalized_wording(self):
        item = self.add_refund(aliases=["退款在哪里？"])
        updated = self.service.add_aliases(item["id"], ["在哪里申请退款", "在哪里申请退款！"])
        repeated = self.service.add_aliases(item["id"], ["退款在哪里!", "在哪里申请退款？", "如何申请退款"])
        self.assertEqual(repeated, updated)
        self.assertEqual(updated["aliases"], ["退款在哪里？", "在哪里申请退款"])
        self.assertEqual(self.service.add_aliases(item["id"], []), updated)
        self.assertEqual(self.service.count(), 1)

    def test_add_aliases_conflict_rolls_back_every_addition_and_releases_keys(self):
        item = self.add_refund(aliases=["用户自己添加的退款问法"])
        self.service.add("人工客服在哪里", "请从帮助中心联系。")
        with self.assertRaises(ValueError):
            self.service.add_aliases(item["id"], ["退款入口在哪里", "人工客服在哪里？"])
        self.assertEqual(self.service.list()[0], item)
        # A failed merge did not reserve its earlier nonconflicting wording.
        extra = self.service.add("退款入口在哪里", "另一条独立答案")
        self.assertNotEqual(extra["id"], item["id"])

    def test_add_aliases_limit_and_invalid_input_do_not_change_original(self):
        item = self.add_refund(aliases=[f"已有问法{i}" for i in range(19)])
        for aliases in (["退款入口在哪里", "在哪里申请退款"], None, "退款在哪里", ["?！"]):
            with self.subTest(aliases=aliases):
                with self.assertRaises(ValueError):
                    self.service.add_aliases(item["id"], aliases)
                self.assertEqual(self.service.list(), [item])
        with self.assertRaisesRegex(ValueError, "不存在"):
            self.service.add_aliases("unknown-faq-id", ["退款在哪里"])
        updated = self.service.add_aliases(item["id"], ["退款在哪里"])
        self.assertEqual(len(updated["aliases"]), 20)

    def test_alias_extension_repairs_only_missing_vectors_and_keyword_index(self):
        item = self.add_refund()
        self.service.match("钱什么时候退回")
        original_vector_calls = sum(map(len, self.embeddings.document_batches))
        other = self.make_service()
        other.add_aliases(item["id"], ["退款入口在哪里"])
        self.service.preview("退款入口在什么地方")
        self.assertEqual(sum(map(len, self.embeddings.document_batches)), original_vector_calls + 1)
        self.assertTrue(any(text == "退款入口在哪里" for _, text in self.service._keyword_phrases))
        collection = self.client.get_collection(self.service.collection_name)
        self.assertEqual(collection.count(), 2)

    def test_location_question_filler_cannot_match_unrelated_customer_service_keywords(self):
        refund = self.add_refund(aliases=["退款入口在哪里"])
        customer = self.service.add("如何联系人工客服", "请打开帮助中心。", ["人工客服在哪里"])
        scores = self.service._keyword_scores(self.service.list(), "退款在哪里")
        self.assertGreater(scores.get(refund["id"], 0), 0)
        self.assertEqual(scores.get(customer["id"], 0), 0)
        tokens = self.service._tokens("退款在哪里，人工客服在哪儿，哪个渠道，营业时间，不支持退款24小时")
        self.assertNotIn("哪里", tokens)
        self.assertNotIn("哪儿", tokens)
        self.assertNotIn("哪个", tokens)
        self.assertNotIn("在", tokens)
        self.assertTrue(any("时间" in token for token in tokens))
        self.assertTrue(any("不" in token for token in tokens))
        self.assertIn("24", tokens)


if __name__ == "__main__":
    unittest.main()
