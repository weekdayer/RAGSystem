"""Small, replaceable FAQ store: SQLite is authoritative; Chroma is a lazy index."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
import threading
import unicodedata
from uuid import uuid4

import jieba
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Plus


MAX_JSON_BYTES = 1024 * 1024
MAX_IMPORT_ITEMS = 500
MAX_QUESTION_LENGTH = 4000
MAX_ANSWER_LENGTH = 12000
MAX_ALIASES = 20

# Only common conversational filler is removed. Negation and numbers stay in the index.
KEYWORD_STOP_WORDS = frozenset({
    "的", "了", "呢", "啊", "呀", "吗", "么", "请", "请问", "我", "我们", "你", "您",
    "一下", "如何", "怎么", "怎样", "什么",
    "在", "哪里", "哪儿", "在哪", "在哪儿", "哪个", "哪", "何处",
})
# The dictionary is read-only after lazy initialization and can be shared by services.
_KEYWORD_TOKENIZER = jieba.Tokenizer()


def normalize_question(text: str) -> str:
    """Ignore typography, preserving words, negation, numbers and numeric separators."""
    text = unicodedata.normalize("NFKC", text).casefold()
    characters = []
    for index, character in enumerate(text):
        numeric_separator = character in ".-+/%" and (
            (index > 0 and text[index - 1].isdigit())
            or (index + 1 < len(text) and text[index + 1].isdigit())
        )
        if character.isspace():
            continue
        if unicodedata.category(character).startswith("P") and not numeric_separator:
            continue
        characters.append(character)
    return "".join(characters)


def _text(value: object, limit: int, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}必须是文字。")
    value = value.strip()
    if not value or len(value) > limit:
        raise ValueError(f"{label}不能为空，且不能超过 {limit} 个字符。")
    if any(unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in value):
        raise ValueError(f"{label}含有不支持的控制字符。")
    return value


def _validate_item(question: object, answer: object, aliases: object = None) -> dict:
    question = _text(question, MAX_QUESTION_LENGTH, "FAQ 标准问题")
    answer = _text(answer, MAX_ANSWER_LENGTH, "FAQ 标准答案")
    normalized = normalize_question(question)
    if not normalized:
        raise ValueError("FAQ 标准问题需要包含有效文字。")
    if aliases is None:
        aliases = []
    if not isinstance(aliases, list) or len(aliases) > MAX_ALIASES:
        raise ValueError(f"FAQ 相似问法必须是列表，且最多 {MAX_ALIASES} 条。")
    cleaned_aliases = []
    seen = {normalized}
    for alias in aliases:
        alias = _text(alias, MAX_QUESTION_LENGTH, "FAQ 相似问法")
        alias_key = normalize_question(alias)
        if not alias_key:
            raise ValueError("FAQ 相似问法需要包含有效文字。")
        if alias_key not in seen:
            cleaned_aliases.append(alias)
            seen.add(alias_key)
    return {"question": question, "answer": answer, "aliases": cleaned_aliases}


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not len(left) or len(left) != len(right):
        return None
    left_norm, right_norm = math.hypot(*left), math.hypot(*right)
    if not left_norm or not right_norm or not math.isfinite(left_norm + right_norm):
        return None
    result = math.fsum(
        (float(a) / left_norm) * (float(b) / right_norm) for a, b in zip(left, right)
    )
    return max(-1.0, min(1.0, result))


class FAQService:
    """Use the same embeddings/client as RAG, with an independent FAQ collection.

    Constructing or changing FAQ content never requests embeddings. Semantic
    searches repair their own index, including changes made by another process.
    A future database/backend can replace this class without changing callers.
    """

    def __init__(
        self,
        database_path: Path,
        *,
        client,
        embeddings: Embeddings,
        collection_name: str,
        threshold: float = 0.70,
        min_margin: float = 0.08,
    ) -> None:
        if (
            not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("FAQ 匹配门槛必须在 0 到 1 之间。")
        if (
            not isinstance(min_margin, (int, float))
            or not math.isfinite(min_margin)
            or not 0 <= min_margin <= 1
        ):
            raise ValueError("FAQ 候选领先差必须在 0 到 1 之间。")
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.threshold = float(threshold)
        self.min_margin = float(min_margin)
        self._index_lock = threading.RLock()
        self._tokenizer = _KEYWORD_TOKENIZER
        self._keyword_signature = None
        self._keyword_index = None
        self._keyword_phrases = []
        with self._database() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS faqs (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    aliases TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS faq_keys (
                    normalized TEXT PRIMARY KEY,
                    faq_id TEXT NOT NULL REFERENCES faqs(id) ON DELETE CASCADE,
                    is_canonical INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS faq_keys_owner ON faq_keys(faq_id);
                CREATE TABLE IF NOT EXISTS faq_settings (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    threshold REAL NOT NULL CHECK(threshold >= 0 AND threshold <= 1),
                    min_margin REAL NOT NULL CHECK(min_margin >= 0 AND min_margin <= 1)
                );
                """
            )

    @contextmanager
    def _database(self):
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _item(row) -> dict:
        return {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "aliases": json.loads(row["aliases"]),
            "updated_at": row["updated_at"],
        }

    def list(self) -> list[dict]:
        with self._database() as connection:
            rows = connection.execute("SELECT * FROM faqs ORDER BY rowid").fetchall()
        return [self._item(row) for row in rows]

    def count(self) -> int:
        with self._database() as connection:
            return connection.execute("SELECT COUNT(*) FROM faqs").fetchone()[0]

    def get_settings(self) -> dict:
        """Read shared saved settings each time; constructor values are only defaults."""
        with self._database() as connection:
            row = connection.execute(
                "SELECT threshold, min_margin FROM faq_settings WHERE singleton = 1"
            ).fetchone()
        if row is None:
            return {"threshold": self.threshold, "min_margin": self.min_margin}
        return {"threshold": float(row["threshold"]), "min_margin": float(row["min_margin"])}

    def update_settings(self, threshold: float, min_margin: float | None = None) -> dict:
        """Persist local FAQ tuning without changing credentials or embedding indexes."""
        for value, label in ((threshold, "FAQ 匹配门槛"), (min_margin, "FAQ 候选领先差")):
            if value is None and label == "FAQ 候选领先差":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{label}必须在 0 到 1 之间。")
        with self._database() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT min_margin FROM faq_settings WHERE singleton = 1"
            ).fetchone()
            margin = (
                float(min_margin) if min_margin is not None
                else float(row["min_margin"]) if row is not None else self.min_margin
            )
            connection.execute(
                "INSERT INTO faq_settings(singleton, threshold, min_margin) VALUES (1, ?, ?) "
                "ON CONFLICT(singleton) DO UPDATE SET "
                "threshold = excluded.threshold, min_margin = excluded.min_margin",
                (float(threshold), margin),
            )
        return {"threshold": float(threshold), "min_margin": margin}

    @staticmethod
    def _equivalent(existing: dict, proposed: dict) -> bool:
        return existing["answer"] == proposed["answer"] and {
            normalize_question(alias) for alias in existing["aliases"]
        } == {normalize_question(alias) for alias in proposed["aliases"]}

    def _insert(self, connection, proposed: dict) -> tuple[dict, bool]:
        canonical = normalize_question(proposed["question"])
        key_row = connection.execute(
            "SELECT faq_id, is_canonical FROM faq_keys WHERE normalized = ?", (canonical,)
        ).fetchone()
        if key_row is not None:
            row = connection.execute(
                "SELECT * FROM faqs WHERE id = ?", (key_row["faq_id"],)
            ).fetchone()
            existing = self._item(row)
            if key_row["is_canonical"] and self._equivalent(existing, proposed):
                return existing, False
            raise ValueError("FAQ 问法已存在或与已有问法冲突；修改内容请先删除原条目。")
        keys = [canonical] + [normalize_question(alias) for alias in proposed["aliases"]]
        for key in keys[1:]:
            if connection.execute(
                "SELECT 1 FROM faq_keys WHERE normalized = ?", (key,)
            ).fetchone():
                raise ValueError("FAQ 相似问法与已有条目冲突，请换一种问法。")
        item = {
            "id": uuid4().hex,
            **proposed,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        }
        connection.execute(
            "INSERT INTO faqs(id, question, answer, aliases, updated_at) VALUES (?, ?, ?, ?, ?)",
            (
                item["id"], item["question"], item["answer"],
                json.dumps(item["aliases"], ensure_ascii=False), item["updated_at"],
            ),
        )
        connection.executemany(
            "INSERT INTO faq_keys(normalized, faq_id, is_canonical) VALUES (?, ?, ?)",
            [(key, item["id"], int(index == 0)) for index, key in enumerate(keys)],
        )
        return item, True

    def add(self, question: str, answer: str, aliases: list[str] | None = None) -> dict:
        proposed = _validate_item(question, answer, aliases)
        with self._database() as connection:
            connection.execute("BEGIN IMMEDIATE")
            item, _ = self._insert(connection, proposed)
        return item

    def add_aliases(self, faq_id: str, aliases: list[str]) -> dict:
        """Atomically extend one FAQ's wording without replacing its identity or answer."""
        if not isinstance(faq_id, str) or not faq_id.strip() or len(faq_id) > 100:
            raise ValueError("FAQ 编号无效。")
        if not isinstance(aliases, list):
            raise ValueError("FAQ 相似问法必须是列表。")
        with self._database() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM faqs WHERE id = ?", (faq_id,)).fetchone()
            if row is None:
                raise ValueError("FAQ 条目不存在。")
            item = self._item(row)
            incoming = _validate_item(item["question"], item["answer"], aliases)["aliases"]
            seen = {normalize_question(item["question"])} | {
                normalize_question(alias) for alias in item["aliases"]
            }
            additions = []
            for alias in incoming:
                key = normalize_question(alias)
                if key in seen:
                    continue
                owner = connection.execute(
                    "SELECT faq_id FROM faq_keys WHERE normalized = ?", (key,)
                ).fetchone()
                if owner is not None and owner["faq_id"] != faq_id:
                    raise ValueError("FAQ 相似问法与已有条目冲突，请换一种问法。")
                additions.append((alias, key))
                seen.add(key)
            if len(item["aliases"]) + len(additions) > MAX_ALIASES:
                raise ValueError(f"每条 FAQ 最多保留 {MAX_ALIASES} 条相似问法。")
            if not additions:
                return item
            item["aliases"] = [*item["aliases"], *(alias for alias, _ in additions)]
            item["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            connection.execute(
                "UPDATE faqs SET aliases = ?, updated_at = ? WHERE id = ?",
                (json.dumps(item["aliases"], ensure_ascii=False), item["updated_at"], faq_id),
            )
            connection.executemany(
                "INSERT INTO faq_keys(normalized, faq_id, is_canonical) VALUES (?, ?, 0)",
                [(key, faq_id) for _, key in additions],
            )
        return item

    def delete(self, faq_id: str) -> bool:
        if not isinstance(faq_id, str) or not faq_id.strip() or len(faq_id) > 100:
            raise ValueError("FAQ 编号无效。")
        with self._database() as connection:
            return connection.execute("DELETE FROM faqs WHERE id = ?", (faq_id,)).rowcount > 0

    def import_json(self, content: bytes) -> dict:
        if not isinstance(content, bytes) or not content or len(content) > MAX_JSON_BYTES:
            raise ValueError("FAQ JSON 文件不能为空，且不能超过 1 MiB。")
        try:
            values = json.loads(content.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("FAQ 文件必须是有效的 UTF-8 JSON。") from exc
        if not isinstance(values, list) or len(values) > MAX_IMPORT_ITEMS:
            raise ValueError(f"FAQ JSON 须为数组，且每次最多导入 {MAX_IMPORT_ITEMS} 条。")
        proposed_items = []
        for value in values:
            if not isinstance(value, dict) or not {"question", "answer"} <= value.keys():
                raise ValueError("每条 FAQ 必须包含 question 和 answer 字段。")
            if value.keys() - {"question", "answer", "aliases"}:
                raise ValueError("FAQ JSON 仅支持 question、answer 和 aliases 字段。")
            proposed_items.append(
                _validate_item(value["question"], value["answer"], value.get("aliases"))
            )
        added = skipped = 0
        with self._database() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for proposed in proposed_items:
                _, created = self._insert(connection, proposed)
                added += int(created)
                skipped += int(not created)
            count = connection.execute("SELECT COUNT(*) FROM faqs").fetchone()[0]
        return {"added": added, "skipped": skipped, "count": count}

    def _exact(self, question: str) -> dict | None:
        with self._database() as connection:
            row = connection.execute(
                "SELECT faqs.* FROM faqs JOIN faq_keys ON faq_keys.faq_id = faqs.id "
                "WHERE faq_keys.normalized = ?",
                (normalize_question(question),),
            ).fetchone()
        return self._item(row) if row is not None else None

    @staticmethod
    def _document(
        item: dict, method: str, score: float, *, bm25_score: float = 0.0,
        hybrid_score: float | None = None,
    ) -> Document:
        return Document(
            page_content=item["answer"],
            metadata={
                "kind": "faq", "faq_id": item["id"], "faq_question": item["question"],
                "source": "FAQ：" + item["question"], "match_method": method,
                "relevance_score": float(score),
                "retrieval_method": "exact" if method == "exact" else "hybrid",
                "bm25_score": float(bm25_score),
                "hybrid_score": float(score if hybrid_score is None else hybrid_score),
            },
        )

    def _sync_index(self, items: list[dict]):
        collection = self.client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        desired = {}
        for item in items:
            for question in [item["question"], *item["aliases"]]:
                content_hash = sha256(question.encode("utf-8")).hexdigest()
                vector_id = item["id"] + "_" + content_hash
                desired[vector_id] = (question, item["id"], content_hash)
        indexed = collection.get(include=["metadatas"])
        existing = dict(zip(indexed["ids"], indexed["metadatas"]))
        stale = [vector_id for vector_id in existing if vector_id not in desired]
        if stale:
            collection.delete(ids=stale)
        missing = [
            (vector_id, text, faq_id, content_hash)
            for vector_id, (text, faq_id, content_hash) in desired.items()
            if (existing.get(vector_id) or {}).get("content_hash") != content_hash
            or (existing.get(vector_id) or {}).get("faq_id") != faq_id
        ]
        # Match Bailian's embedding batch limit; failed batches are repaired next time.
        for offset in range(0, len(missing), 10):
            batch = missing[offset:offset + 10]
            vectors = self.embeddings.embed_documents([row[1] for row in batch])
            if len(vectors) != len(batch) or any(_cosine(v, v) is None for v in vectors):
                raise ValueError("FAQ 向量模型没有返回有效向量，请稍后重试。")
            collection.upsert(
                ids=[row[0] for row in batch], embeddings=vectors,
                documents=[row[1] for row in batch],
                metadatas=[{"faq_id": row[2], "content_hash": row[3]} for row in batch],
            )
        return collection

    def _tokens(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFKC", text).casefold()
        return [
            token for token in self._tokenizer.lcut_for_search(text, HMM=False)
            if token not in KEYWORD_STOP_WORDS and any(character.isalnum() for character in token)
        ]

    def _keyword_scores(self, items: list[dict], question: str) -> dict[str, float]:
        signature = tuple(
            (item["id"], item["question"], tuple(item["aliases"])) for item in items
        )
        if signature != self._keyword_signature:
            phrases = [
                (item["id"], text) for item in items
                for text in [item["question"], *item["aliases"]]
            ]
            corpus = [self._tokens(text) for _, text in phrases]
            # Positive IDF works with tiny FAQ collections; delta=0 gives zero to nonmatches.
            self._keyword_index = BM25Plus(corpus, delta=0) if any(corpus) else None
            self._keyword_phrases = phrases
            self._keyword_signature = signature
        if self._keyword_index is None:
            return {}
        scores = {}
        for (faq_id, _), raw_score in zip(
            self._keyword_phrases, self._keyword_index.get_scores(self._tokens(question))
        ):
            score = float(raw_score)
            if math.isfinite(score) and score > 0:
                scores[faq_id] = max(score, scores.get(faq_id, 0.0))
        return scores

    def _collect_candidates(
        self, question: str, items: list[dict]
    ) -> tuple[list[dict], dict[str, dict]]:
        """Merge vector and keyword candidates, keeping each FAQ's best scores."""
        with self._index_lock:
            collection = self._sync_index(items)
            keyword_scores = self._keyword_scores(items, question)
            query_vector = self.embeddings.embed_query(question)
            if _cosine(query_vector, query_vector) is None:
                return [], {}
            vector_count = collection.count()
            if not vector_count:
                return [], {}
            results = collection.query(
                query_embeddings=[query_vector], n_results=min(vector_count, MAX_ALIASES + 2),
                include=["metadatas", "embeddings"],
            )
            if results["embeddings"] is None:
                return [], {}
            vector_candidates = dict(zip(
                results["ids"][0],
                zip(results["metadatas"][0], results["embeddings"][0]),
            ))
            if keyword_scores:
                # Union keyword candidates with vector candidates. Fetch every alias of
                # those FAQs so their strongest semantic score cannot be hidden by ANN.
                lexical = collection.get(
                    where={"faq_id": {"$in": list(keyword_scores)}},
                    include=["metadatas", "embeddings"],
                )
                if lexical["embeddings"] is not None:
                    vector_candidates.update(zip(
                        lexical["ids"], zip(lexical["metadatas"], lexical["embeddings"]),
                    ))
            # Re-read SQLite so a concurrently deleted answer can never be returned.
            current_items = {item["id"]: item for item in self.list()}
            scores = {}
            for metadata, vector in vector_candidates.values():
                faq_id = (metadata or {}).get("faq_id")
                if faq_id not in current_items:
                    continue
                score = _cosine(query_vector, vector)
                if score is not None:
                    scores[faq_id] = max(score, scores.get(faq_id, -1.0))
        candidates = [
            {
                "id": faq_id, "question": current_items[faq_id]["question"],
                "semantic_score": score, "bm25_score": keyword_scores.get(faq_id, 0.0),
            }
            for faq_id, score in scores.items()
        ]
        return candidates, current_items

    @staticmethod
    def _rank_candidates(candidates: list[dict]) -> list[dict]:
        """Combine semantic and normalized BM25 scores, then rank complete FAQs."""
        keyword_max = max((item["bm25_score"] for item in candidates), default=0.0)
        ranked = []
        for candidate in candidates:
            # With no keyword overlap, preserve the existing semantic margin behavior.
            hybrid_score = (
                0.8 * candidate["semantic_score"] + 0.2 * candidate["bm25_score"] / keyword_max
                if keyword_max > 0 else candidate["semantic_score"]
            )
            ranked.append({**candidate, "hybrid_score": hybrid_score})
        ranked.sort(key=lambda item: (item["hybrid_score"], item["semantic_score"]), reverse=True)
        return ranked

    @staticmethod
    def _decide_match(candidates: list[dict], items: dict[str, dict], settings: dict) -> dict:
        """Apply the semantic threshold and winning margin to the top hybrid result."""
        decision = {"matched": False, "reason": "no_candidates", "match": None}
        if not candidates:
            return decision
        winner = candidates[0]
        # Gate the overall hybrid winner; never discard a low-semantic keyword hit
        # and then force a different FAQ merely because its cosine is higher.
        if (
            winner["semantic_score"] < settings["threshold"]
            and not math.isclose(
                winner["semantic_score"], settings["threshold"], rel_tol=0.0, abs_tol=1e-7
            )
        ):
            decision["reason"] = "below_threshold"
            return decision
        if (
            len(candidates) > 1
            and winner["hybrid_score"] - candidates[1]["hybrid_score"] < settings["min_margin"]
        ):
            decision["reason"] = "ambiguous"
            return decision
        item = items[winner["id"]]
        decision.update(
            matched=True, reason="hybrid",
            match={
                "id": item["id"], "question": item["question"], "answer": item["answer"],
                "match_method": "semantic",
            },
        )
        return decision

    def preview(self, question: str) -> dict:
        """Explain the same exact/hybrid decision that normal chat will use.

        Semantic and hybrid scores are on a -1..1 scale; BM25 is a raw keyword
        ranking score, not a percentage or confidence probability.
        """
        question = _text(question, MAX_QUESTION_LENGTH, "问题")
        settings = self.get_settings()
        result = {
            "question": question, "matched": False, "reason": "no_candidates",
            **settings, "match": None, "candidates": [],
        }
        if not normalize_question(question):
            return result
        exact = self._exact(question)
        if exact is not None:
            result.update(
                matched=True, reason="exact",
                match={
                    "id": exact["id"], "question": exact["question"], "answer": exact["answer"],
                    "match_method": "exact",
                },
                candidates=[{
                    "id": exact["id"], "question": exact["question"], "semantic_score": 1.0,
                    "bm25_score": 0.0, "hybrid_score": 1.0,
                }],
            )
            return result
        items = self.list()
        if not items:
            result["reason"] = "empty"
            return result
        candidates, current_items = self._collect_candidates(question, items)
        result["candidates"] = self._rank_candidates(candidates)
        result.update(self._decide_match(result["candidates"], current_items, settings))
        return result

    def match(self, question: str) -> Document | None:
        decision = self.preview(question)
        if not decision["matched"]:
            return None
        match = decision["match"]
        candidate = next(item for item in decision["candidates"] if item["id"] == match["id"])
        return self._document(
            match, match["match_method"], candidate["semantic_score"],
            bm25_score=candidate["bm25_score"], hybrid_score=candidate["hybrid_score"],
        )
