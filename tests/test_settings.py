"""配置入口测试：隔离本机环境，不读取真实密钥或创建知识库。"""

from contextlib import chdir
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from settings import DEFAULT_BASE_URL, ROOT, RagSettings, load_rag_settings


class SettingsTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for name in (
            "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "CHAT_MODEL", "EMBEDDING_MODEL",
            "CHROMA_PERSIST_DIRECTORY", "RAG_RELEVANCE_THRESHOLD", "RAG_TOP_K",
            "FAQ_MATCH_THRESHOLD", "FAQ_MIN_MARGIN",
        ):
            os.environ.pop(name, None)
        env_file = patch("settings.load_environment")
        env_file.start()
        self.addCleanup(env_file.stop)

    def test_defaults_allow_missing_key_and_empty_model_fields(self):
        for empty_values in ({}, {"DASHSCOPE_BASE_URL": "", "CHAT_MODEL": "", "EMBEDDING_MODEL": ""}):
            with self.subTest(empty_values=empty_values), patch.dict(os.environ, empty_values):
                settings = load_rag_settings()
                self.assertEqual(settings.api_key, "")
                self.assertEqual(settings.base_url, DEFAULT_BASE_URL)
                self.assertEqual(settings.chat_model, "qwen-plus")
                self.assertEqual(settings.embedding_model, "text-embedding-v4")
                self.assertEqual(settings.top_k, 5)
                self.assertEqual(settings.relevance_threshold, 0.35)

    def test_environment_controls_shared_model_and_retrieval_settings(self):
        os.environ.update(
            DASHSCOPE_API_KEY="local-test-key", DASHSCOPE_BASE_URL="https://example.invalid/v1/",
            CHAT_MODEL="test-chat", EMBEDDING_MODEL="test-embedding", RAG_TOP_K="7",
            RAG_RELEVANCE_THRESHOLD="0.6", FAQ_MATCH_THRESHOLD="0.8", FAQ_MIN_MARGIN="0.1",
        )
        settings = load_rag_settings()
        self.assertEqual(settings.base_url, "https://example.invalid/v1")
        self.assertEqual((settings.chat_model, settings.embedding_model), ("test-chat", "test-embedding"))
        self.assertEqual((settings.top_k, settings.relevance_threshold), (7, 0.6))
        self.assertEqual((settings.faq_threshold, settings.faq_min_margin), (0.8, 0.1))
        self.assertNotIn("local-test-key", repr(settings))

    def test_empty_storage_path_uses_project_data_directory(self):
        for value in ("", " \t "):
            with self.subTest(value=value):
                os.environ["CHROMA_PERSIST_DIRECTORY"] = value
                self.assertEqual(load_rag_settings().persist_directory, ROOT / "data" / "chroma")

    def test_relative_storage_path_is_independent_of_working_directory(self):
        os.environ["CHROMA_PERSIST_DIRECTORY"] = "test-data/chroma"
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            self.assertEqual(load_rag_settings().persist_directory, ROOT / "test-data" / "chroma")

    def test_absolute_storage_path_keeps_its_target(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["CHROMA_PERSIST_DIRECTORY"] = directory
            self.assertEqual(load_rag_settings().persist_directory, Path(directory).resolve())

    def test_invalid_numeric_configuration_does_not_echo_its_value(self):
        for name in ("RAG_TOP_K", "RAG_RELEVANCE_THRESHOLD", "FAQ_MATCH_THRESHOLD", "FAQ_MIN_MARGIN"):
            with self.subTest(name=name), patch.dict(os.environ, {name: "private-invalid-value"}):
                with self.assertRaises(ValueError) as raised:
                    load_rag_settings()
                self.assertNotIn("private-invalid-value", str(raised.exception))

    def test_settings_can_be_imported_without_loading_the_engine(self):
        code = (
            "import settings, sys, json; "
            "print(json.dumps([name for name in sys.modules "
            "if name == 'rag' or name.startswith(('chromadb', 'langchain'))]))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
        )
        self.assertEqual(json.loads(result.stdout), [])

    def test_old_rag_settings_import_remains_compatible(self):
        from rag import RagSettings as PreviousImport

        self.assertIs(PreviousImport, RagSettings)


if __name__ == "__main__":
    unittest.main()
