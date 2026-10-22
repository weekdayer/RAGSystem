"""语音适配器的离线协议测试：只使用内存音频和 MockTransport。"""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
import struct
import traceback
import unittest
from unittest.mock import Mock, patch
import wave

import httpx

from speech import (
    BailianTranscriber,
    DEFAULT_ASR_MODEL,
    DEFAULT_BASE_URL,
    HttpTranscriber,
    MAX_AUDIO_BYTES,
    SpeechServiceError,
    SpeechToText,
    create_transcriber,
)


def make_wav(seconds: float = 1, *, rate: int = 16000) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(rate)
        recording.writeframes(b"\x01\x00" * int(seconds * rate))
    return output.getvalue()


def chat_response(content: object = "你好，这是测试。") -> dict:
    return {
        "id": "chatcmpl-asr-offline",
        "object": "chat.completion",
        "created": 0,
        "model": DEFAULT_ASR_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


class SpeechTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audio = make_wav()
        self.requests: list[httpx.Request] = []

    def transport_client(self, handler=None) -> httpx.Client:
        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if handler is not None:
                return handler(request)
            return httpx.Response(200, json=chat_response())

        client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(client.close)
        return client

    def bailian(self, handler=None) -> BailianTranscriber:
        return BailianTranscriber(
            api_key="offline-test-key",
            base_url="https://asr.example.invalid/compatible-mode/v1",
            http_client=self.transport_client(handler),
            max_retries=0,
        )

    def lab(self, handler=None, *, api_key: str = "") -> HttpTranscriber:
        if handler is None:
            handler = lambda request: httpx.Response(200, json={"text": "  实验室识别结果。  "})
        return HttpTranscriber(
            url="http://lab.example.invalid/transcribe",
            api_key=api_key,
            client=self.transport_client(handler),
        )

    def test_bailian_request_uses_audio_data_uri_and_chat_endpoint(self) -> None:
        transcriber = self.bailian()
        self.assertIsInstance(transcriber, SpeechToText)
        self.assertEqual(transcriber.transcribe(self.audio), "你好，这是测试。")
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request.url.host, "asr.example.invalid")
        self.assertEqual(request.url.path, "/compatible-mode/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer offline-test-key")
        body = json.loads(request.content)
        self.assertEqual(body["model"], DEFAULT_ASR_MODEL)
        self.assertIs(body["stream"], False)
        self.assertNotIn("stream_options", body)
        self.assertEqual(body["asr_options"], {"enable_itn": False})
        self.assertEqual(len(body["messages"]), 1)
        message = body["messages"][0]
        self.assertEqual(message["role"], "user")
        self.assertEqual(len(message["content"]), 1)
        part = message["content"][0]
        self.assertEqual(part["type"], "input_audio")
        prefix, encoded = part["input_audio"]["data"].split(",", 1)
        self.assertEqual(prefix, "data:audio/wav;base64")
        self.assertEqual(base64.b64decode(encoded), self.audio)

    def test_bailian_allows_injected_openai_client_and_custom_model(self) -> None:
        client = Mock()
        client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=" 自定义模型。 "))]
        )
        transcriber = BailianTranscriber(
            api_key="offline-key", model="custom-asr-model", client=client
        )
        self.assertEqual(transcriber.transcribe(self.audio), "自定义模型。")
        self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], "custom-asr-model")

    def test_bailian_normalizes_mime_aliases_and_parameters(self) -> None:
        transcriber = self.bailian()
        transcriber.transcribe(
            self.audio, filename="RECORDING.WAV", mime_type="Audio/X-Wav; rate=16000"
        )
        body = json.loads(self.requests[0].content)
        self.assertTrue(body["messages"][0]["content"][0]["input_audio"]["data"].startswith("data:audio/wav;"))

    def test_common_compressed_formats_are_forwarded_without_transcoding(self) -> None:
        for name, mime in (
            ("speech.mp3", "audio/mpeg"),
            ("speech.flac", "audio/flac"),
            ("speech.ogg", "audio/ogg"),
            ("speech.webm", "audio/webm"),
        ):
            with self.subTest(name=name):
                transcriber = self.bailian()
                transcriber.transcribe(b"offline-compressed-audio", filename=name, mime_type=mime)
                body = json.loads(self.requests[-1].content)
                data_uri = body["messages"][0]["content"][0]["input_audio"]["data"]
                self.assertTrue(data_uri.startswith(f"data:{mime};base64,"))

    def test_bailian_rejects_m4a_but_http_adapter_can_forward_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "M4A"):
            self.bailian().transcribe(b"offline-m4a", filename="voice.m4a", mime_type="audio/mp4")
        self.assertEqual(len(self.requests), 0)
        self.assertEqual(
            self.lab().transcribe(b"offline-m4a", filename="voice.m4a", mime_type="audio/mp4"),
            "实验室识别结果。",
        )

    def test_http_lab_uses_multipart_file_and_optional_bearer_key(self) -> None:
        transcriber = self.lab(api_key="lab-offline-key")
        self.assertIsInstance(transcriber, SpeechToText)
        transcript = transcriber.transcribe(
            self.audio, filename=r"C:\private\voice.wav", mime_type="audio/wav"
        )
        self.assertEqual(transcript, "实验室识别结果。")
        request = self.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.url.path, "/transcribe")
        self.assertEqual(request.headers["authorization"], "Bearer lab-offline-key")
        self.assertTrue(request.headers["content-type"].startswith("multipart/form-data;"))
        body = request.content
        self.assertIn(b'name="file"; filename="voice.wav"', body)
        self.assertIn(b"Content-Type: audio/wav", body)
        self.assertIn(self.audio, body)
        self.assertNotIn(b"private", body)

    def test_http_lab_omits_authorization_without_key(self) -> None:
        self.lab().transcribe(self.audio, filename="../../voice.wav")
        self.assertNotIn("authorization", self.requests[0].headers)
        self.assertIn(b'filename="voice.wav"', self.requests[0].content)
        self.assertNotIn(b"../", self.requests[0].content)

    def test_invalid_audio_fails_before_any_request(self) -> None:
        invalid_cases = (
            (b"", "empty.wav", "audio/wav", "为空"),
            (b"not-wave", "broken.wav", "audio/wav", "WAV 文件头"),
            (self.audio, "recording.exe", "audio/wav", "支持"),
            (self.audio, "recording.wav", "application/octet-stream", "类型"),
            (self.audio, "recording.wav", "audio/mpeg", "类型"),
            (self.audio, "voice\n.wav", "audio/wav", "文件名"),
        )
        for constructor in (self.bailian, self.lab):
            for audio, filename, mime, expected in invalid_cases:
                with self.subTest(adapter=constructor.__name__, filename=filename):
                    with self.assertRaisesRegex(ValueError, expected):
                        constructor().transcribe(audio, filename=filename, mime_type=mime)
        self.assertEqual(self.requests, [])

    def test_raw_audio_larger_than_ten_mb_is_rejected(self) -> None:
        audio = b"a" * (MAX_AUDIO_BYTES + 1)
        for constructor in (self.bailian, self.lab):
            with self.subTest(adapter=constructor.__name__):
                with self.assertRaisesRegex(ValueError, "10 MB"):
                    constructor().transcribe(audio, filename="big.mp3", mime_type="audio/mpeg")
        self.assertEqual(self.requests, [])

    def test_bailian_also_checks_base64_encoded_limit(self) -> None:
        audio = b"a" * (MAX_AUDIO_BYTES * 3 // 4)
        with self.assertRaisesRegex(ValueError, "编码后"):
            self.bailian().transcribe(audio, filename="big.mp3", mime_type="audio/mpeg")
        self.assertEqual(self.requests, [])

    def test_wav_longer_than_five_minutes_is_rejected(self) -> None:
        audio = make_wav(301, rate=10)
        for constructor in (self.bailian, self.lab):
            with self.subTest(adapter=constructor.__name__):
                with self.assertRaisesRegex(ValueError, "5 分钟"):
                    constructor().transcribe(audio)
        self.assertEqual(self.requests, [])

    def test_wav_exactly_five_minutes_is_within_duration_limit(self) -> None:
        self.assertEqual(self.bailian().transcribe(make_wav(300, rate=10)), "你好，这是测试。")

    def test_wav_with_no_samples_or_truncated_samples_is_rejected(self) -> None:
        for audio, expected in ((make_wav(0), "没有音频采样"), (self.audio[:-100], "内容不完整")):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    self.bailian().transcribe(audio)
        self.assertEqual(self.requests, [])

    def test_wav_with_invalid_chunk_length_is_rejected_without_parser_traceback(self) -> None:
        audio = self.audio[:16] + struct.pack("<I", 0xFFFFFFFF) + self.audio[20:]
        with self.assertRaisesRegex(ValueError, "WAV 文件头"):
            self.bailian().transcribe(audio)
        self.assertEqual(self.requests, [])

    def test_invalid_bailian_transcript_is_not_returned(self) -> None:
        for content in (None, "", " \n\t", ["unexpected"], "你好\x00世界"):
            with self.subTest(content=content):
                handler = lambda request: httpx.Response(200, json=chat_response(content))
                with self.assertRaises(ValueError):
                    self.bailian(handler).transcribe(self.audio)

    def test_missing_bailian_choices_is_readable(self) -> None:
        payload = chat_response()
        payload["choices"] = []
        with self.assertRaisesRegex(ValueError, "文字格式无效"):
            self.bailian(lambda request: httpx.Response(200, json=payload)).transcribe(self.audio)

    def test_http_requires_json_object_with_valid_text(self) -> None:
        for payload in ({}, [], {"text": None}, {"text": 123}, {"text": " "}, {"text": "bad\x00text"}):
            with self.subTest(payload=payload):
                handler = lambda request: httpx.Response(200, json=payload)
                with self.assertRaises(ValueError):
                    self.lab(handler).transcribe(self.audio)
        with self.assertRaisesRegex(ValueError, "JSON"):
            self.lab(lambda request: httpx.Response(200, text="not json")).transcribe(self.audio)

    def test_remote_errors_never_include_response_or_key_in_traceback(self) -> None:
        secret = "private-key-must-not-appear"
        for constructor in (self.bailian, self.lab):
            for status in (400, 401, 403, 404, 413, 429, 500):
                with self.subTest(adapter=constructor.__name__, status=status):
                    handler = lambda request: httpx.Response(
                        status,
                        json={"error": {"message": secret, "code": secret}, "text": secret},
                    )
                    try:
                        constructor(handler).transcribe(self.audio)
                    except SpeechServiceError as error:
                        rendered = "".join(traceback.format_exception(error))
                        self.assertNotIn(secret, str(error))
                        self.assertNotIn(secret, rendered)
                    else:
                        self.fail("Expected a sanitized service error")

    def test_connection_and_timeout_errors_are_readable_and_sanitized(self) -> None:
        secret = "sensitive-audio-and-token"
        for error_type, expected in ((httpx.ConnectError, "无法连接"), (httpx.ReadTimeout, "超时")):
            def handler(request, error_type=error_type):
                raise error_type(secret, request=request)

            for constructor in (self.bailian, self.lab):
                with self.subTest(adapter=constructor.__name__, error=error_type.__name__):
                    with self.assertRaisesRegex(SpeechServiceError, expected) as raised:
                        constructor(handler).transcribe(self.audio)
                    self.assertNotIn(secret, str(raised.exception))

    def test_http_redirect_does_not_forward_audio_to_another_server(self) -> None:
        handler = lambda request: httpx.Response(307, headers={"Location": "https://other.invalid"})
        with self.assertRaisesRegex(SpeechServiceError, "重定向"):
            self.lab(handler).transcribe(self.audio)
        self.assertEqual(len(self.requests), 1)

    def test_invalid_configuration_does_not_echo_values(self) -> None:
        secret = "private-setting-value"
        for url in ("", secret, f"https://user:{secret}@example.invalid/path", f"ftp://{secret}"):
            with self.subTest(url_kind=url.split(":", 1)[0]):
                for constructor in (
                    lambda: HttpTranscriber(url=url),
                    lambda: BailianTranscriber(api_key="offline-key", base_url=url),
                ):
                    with self.assertRaises(ValueError) as raised:
                        constructor()
                    self.assertNotIn(secret, str(raised.exception))
        with self.assertRaisesRegex(ValueError, "ASR_API_KEY"):
            BailianTranscriber(api_key=" ")


class SpeechFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.loader = patch("speech.load_environment")
        self.load_environment = self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_default_factory_uses_chat_environment_without_network(self) -> None:
        os.environ["DASHSCOPE_API_KEY"] = "offline-chat-key"
        with patch("speech.BailianTranscriber") as adapter:
            self.assertIs(create_transcriber(), adapter.return_value)
        adapter.assert_called_once_with(
            api_key="offline-chat-key", base_url=DEFAULT_BASE_URL, model=DEFAULT_ASR_MODEL
        )
        self.load_environment.assert_called_once_with()

    def test_separate_asr_configuration_takes_priority(self) -> None:
        os.environ.update(
            {
                "ASR_PROVIDER": " BAILIAN ",
                "ASR_API_KEY": "asr-test-key",
                "ASR_BASE_URL": "https://separate.invalid/compatible-mode/v1",
                "ASR_MODEL": "custom-test-asr",
                "DASHSCOPE_API_KEY": "chat-test-key",
                "DASHSCOPE_BASE_URL": "https://chat.invalid/v1",
            }
        )
        with patch("speech.BailianTranscriber") as adapter:
            create_transcriber(fallback_api_key="ui-key", fallback_base_url="https://ui.invalid/v1")
        adapter.assert_called_once_with(
            api_key="asr-test-key",
            base_url="https://separate.invalid/compatible-mode/v1",
            model="custom-test-asr",
        )

    def test_ui_fallback_takes_priority_over_chat_environment(self) -> None:
        os.environ.update(
            {"DASHSCOPE_API_KEY": "environment-key", "DASHSCOPE_BASE_URL": "https://env.invalid/v1"}
        )
        with patch("speech.BailianTranscriber") as adapter:
            create_transcriber(fallback_api_key=" ui-key ", fallback_base_url=" https://ui.invalid/v1 ")
        adapter.assert_called_once_with(
            api_key="ui-key", base_url="https://ui.invalid/v1", model=DEFAULT_ASR_MODEL
        )

    def test_http_factory_uses_independent_lab_protocol(self) -> None:
        os.environ.update(
            {"ASR_PROVIDER": "http", "ASR_HTTP_URL": "http://lab.invalid/asr", "ASR_HTTP_API_KEY": "lab-key"}
        )
        with patch("speech.HttpTranscriber") as adapter:
            self.assertIs(create_transcriber(fallback_api_key="chat-key"), adapter.return_value)
        adapter.assert_called_once_with(url="http://lab.invalid/asr", api_key="lab-key")

    def test_missing_or_unknown_factory_configuration_is_sanitized(self) -> None:
        with self.assertRaisesRegex(ValueError, "ASR_API_KEY"):
            create_transcriber()
        os.environ["ASR_PROVIDER"] = "http"
        with self.assertRaisesRegex(ValueError, "ASR_HTTP_URL"):
            create_transcriber()
        os.environ["ASR_PROVIDER"] = "secret-provider-value"
        with self.assertRaises(ValueError) as raised:
            create_transcriber()
        self.assertNotIn("secret-provider-value", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
