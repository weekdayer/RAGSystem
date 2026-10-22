"""可替换的语音转文字适配器；音频仅在内存中处理，不写入磁盘。"""

from __future__ import annotations

import base64
from io import BytesIO
import os
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit
import wave

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from settings import load_environment


MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_AUDIO_SECONDS = 5 * 60
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ASR_MODEL = "qwen3-asr-flash"

_AUDIO_FORMATS = {
    ".wav": ("audio/wav", {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}),
    ".mp3": ("audio/mpeg", {"audio/mpeg", "audio/mp3", "audio/x-mp3"}),
    ".flac": ("audio/flac", {"audio/flac", "audio/x-flac"}),
    ".ogg": ("audio/ogg", {"audio/ogg", "application/ogg"}),
    ".m4a": ("audio/mp4", {"audio/mp4", "audio/m4a", "audio/x-m4a"}),
    ".webm": ("audio/webm", {"audio/webm", "video/webm"}),
}


class SpeechServiceError(ValueError):
    """可直接展示的固定语音服务错误；与本地输入校验错误区分。"""


@runtime_checkable
class SpeechToText(Protocol):
    """实验室 ASR 可以实现同一方法，页面无需依赖某个供应商。"""

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "recording.wav",
        mime_type: str = "audio/wav",
    ) -> str: ...


def _validate_url(value: str, *, setting: str) -> str:
    """只返回固定配置提示，避免错误中带出 URL 凭证或查询参数。"""
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and parsed.port != 0
            and not parsed.fragment
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"请将 {setting} 配置为不含用户名或密码的完整 HTTP(S) 地址。")
    return value


def _validate_audio(audio: bytes, filename: str, mime_type: str) -> tuple[bytes, str, str]:
    if not isinstance(audio, (bytes, bytearray)) or not audio:
        raise ValueError("音频内容为空，请先录音或选择音频文件。")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("音频超过 10 MB，请缩短录音或压缩后重试。")
    if not isinstance(filename, str) or not isinstance(mime_type, str):
        raise ValueError("音频文件名或格式无效，请重新选择音频文件。")
    # 同时处理浏览器和 Windows 路径，远端仅收到文件名。
    name = PurePosixPath(filename.replace("\\", "/")).name
    if not name or any(character in name for character in ("\x00", "\r", "\n")):
        raise ValueError("音频文件名无效，请重新选择音频文件。")
    extension = PurePosixPath(name).suffix.lower()
    if extension not in _AUDIO_FORMATS:
        raise ValueError("暂时只支持 WAV、MP3、FLAC、OGG、M4A 和 WebM 音频。")
    canonical_mime, allowed_mimes = _AUDIO_FORMATS[extension]
    if mime_type.split(";", 1)[0].strip().lower() not in allowed_mimes:
        raise ValueError("音频类型不受支持或与扩展名不一致，请重新选择音频文件。")
    content = bytes(audio)
    if extension == ".wav":
        try:
            with wave.open(BytesIO(content), "rb") as recording:
                frames = recording.getnframes()
                rate = recording.getframerate()
                if rate <= 0:
                    raise wave.Error("invalid sample rate")
                if frames <= 0:
                    raise ValueError("录音中没有音频采样，请重新录音。")
                if frames / rate > MAX_AUDIO_SECONDS:
                    raise ValueError("音频超过 5 分钟，请缩短录音后重试。")
                expected_size = frames * recording.getnchannels() * recording.getsampwidth()
                if len(recording.readframes(frames)) < expected_size:
                    raise ValueError("WAV 音频内容不完整，请重新录音或选择未损坏的文件。")
        except (wave.Error, EOFError, RuntimeError):
            raise ValueError("无法读取 WAV 文件头，请重新录音或转换为 PCM WAV。") from None
    return content, name, canonical_mime


def _validate_transcript(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("语音服务返回的文字格式无效，请检查服务的响应协议。")
    text = value.strip()
    if not text:
        raise ValueError("没有识别到文字，请靠近麦克风重新录音。")
    if any(ord(character) < 32 and character not in "\n\r\t" for character in text):
        raise ValueError("语音服务返回了无效文字，请重新录音或检查服务。")
    return text


def _remote_error_message(error: Exception) -> str:
    """不引用远端响应、请求头、音频数据或可能含秘密的异常正文。"""
    if isinstance(error, (APITimeoutError, httpx.TimeoutException)):
        return "语音识别请求超时，请稍后重试或缩短录音。"
    if isinstance(error, (APIConnectionError, httpx.RequestError)):
        return "无法连接语音服务，请检查网络和语音服务地址。"
    if isinstance(error, APIStatusError):
        status = error.status_code
    elif isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
    else:
        return "语音服务请求失败，请检查服务配置后重试。"
    if 300 <= status < 400:
        return "语音接口发生重定向，请将 ASR_HTTP_URL 配置为最终接口地址。"
    messages = {
        400: "语音服务未接受音频，请检查模型、音频格式及所属地域，或转换为 WAV 重试。",
        401: "语音服务认证失败，请检查 API Key 及其所属地域。",
        403: "没有语音服务调用权限，请检查模型开通状态、额度和地域。",
        404: "未找到语音接口或模型，请检查语音服务地址和模型名称。",
        413: "语音服务拒绝了过大的音频，请缩短录音或压缩后重试。",
        429: "语音服务请求过于频繁或额度不足，请稍后重试。",
    }
    return messages.get(status, "语音服务暂时不可用，请稍后重试。")


class BailianTranscriber:
    """百炼 Qwen-ASR 的 OpenAI 兼容协议，使用内存中的 Base64 Data URL。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_ASR_MODEL,
        client: OpenAI | None = None,
        http_client: httpx.Client | None = None,
        timeout: float = 90,
        max_retries: int = 1,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("请配置 ASR_API_KEY 或 DASHSCOPE_API_KEY 后再使用语音输入。")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("请配置有效的 ASR_MODEL 模型名称。")
        self.model = model.strip()
        normalized_url = _validate_url(base_url, setting="ASR_BASE_URL").rstrip("/")
        try:
            self.client = client if client is not None else OpenAI(
                api_key=api_key.strip(),
                base_url=normalized_url,
                timeout=timeout,
                max_retries=max_retries,
                http_client=http_client,
            )
        except Exception:
            raise ValueError("语音服务配置无效，请检查 API Key 和调用地址。") from None

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "recording.wav",
        mime_type: str = "audio/wav",
    ) -> str:
        content, name, canonical_mime = _validate_audio(audio, filename, mime_type)
        if PurePosixPath(name).suffix.lower() == ".m4a":
            raise ValueError("当前百炼语音适配器暂不支持 M4A，请转换为 WAV 或 MP3 后重试。")
        prefix = f"data:{canonical_mime};base64,"
        # 百炼限制编码后的输入也不超过 10 MB，原文件约需小于 7.5 MB。
        if len(prefix) + 4 * ((len(content) + 2) // 3) > MAX_AUDIO_BYTES:
            raise ValueError("音频编码后超过百炼的 10 MB 限制，请将原文件缩小到约 7.5 MB 以下。")
        data_uri = prefix + base64.b64encode(content).decode("ascii")
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                stream=False,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_audio", "input_audio": {"data": data_uri}}
                        ],
                    }
                ],
                extra_body={"asr_options": {"enable_itn": False}},
            )
        except Exception as error:
            raise SpeechServiceError(_remote_error_message(error)) from None
        try:
            transcript = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            raise ValueError("语音服务返回的文字格式无效，请检查服务的响应协议。") from None
        return _validate_transcript(transcript)


class HttpTranscriber:
    """实验室接口：multipart 的 file 字段上传音频，JSON 返回 {\"text\": \"...\"}。"""

    def __init__(
        self,
        *,
        url: str,
        api_key: str = "",
        client: httpx.Client | None = None,
        timeout: float = 90,
        max_retries: int = 1,
    ) -> None:
        self.url = _validate_url(url, setting="ASR_HTTP_URL")
        if not isinstance(api_key, str):
            raise ValueError("ASR_HTTP_API_KEY 配置无效。")
        self._api_key = api_key.strip()
        self._client = client
        self.timeout = timeout
        self.max_retries = max_retries

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "recording.wav",
        mime_type: str = "audio/wav",
    ) -> str:
        content, name, canonical_mime = _validate_audio(audio, filename, mime_type)
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        request_arguments = {
            "files": {"file": (name, content, canonical_mime)},
            "headers": headers,
            "timeout": self.timeout,
            "follow_redirects": False,
        }
        try:
            if self._client is not None:
                response = self._client.post(self.url, **request_arguments)
            else:
                # 只重试建立连接；不自动重复成功提交的音频或 5xx 请求。
                transport = httpx.HTTPTransport(retries=self.max_retries)
                with httpx.Client(transport=transport, timeout=self.timeout) as client:
                    response = client.post(self.url, **request_arguments)
            response.raise_for_status()
        except ValueError:
            raise ValueError("语音服务地址或请求配置无效，请检查 ASR_HTTP_URL。") from None
        except Exception as error:
            raise SpeechServiceError(_remote_error_message(error)) from None
        try:
            payload = response.json()
        except (ValueError, UnicodeDecodeError):
            raise ValueError("语音服务未返回有效 JSON，请按协议返回 text 字段。") from None
        if not isinstance(payload, dict) or "text" not in payload:
            raise ValueError("语音服务响应缺少 text 字段，请检查实验室接口协议。")
        return _validate_transcript(payload["text"])


def create_transcriber(
    *,
    fallback_api_key: str | None = None,
    fallback_base_url: str | None = None,
) -> SpeechToText:
    """根据 ASR_PROVIDER 选择服务；仅加载配置，不在构造时发送音频或请求。"""
    load_environment()
    provider = os.getenv("ASR_PROVIDER", "bailian").strip().lower() or "bailian"
    if provider == "bailian":
        return BailianTranscriber(
            api_key=os.getenv("ASR_API_KEY", "").strip()
            or (fallback_api_key or "").strip()
            or os.getenv("DASHSCOPE_API_KEY", "").strip(),
            base_url=os.getenv("ASR_BASE_URL", "").strip()
            or (fallback_base_url or "").strip()
            or os.getenv("DASHSCOPE_BASE_URL", "").strip()
            or DEFAULT_BASE_URL,
            model=os.getenv("ASR_MODEL", "").strip() or DEFAULT_ASR_MODEL,
        )
    if provider == "http":
        return HttpTranscriber(
            url=os.getenv("ASR_HTTP_URL", "").strip(),
            api_key=os.getenv("ASR_HTTP_API_KEY", "").strip(),
        )
    raise ValueError("ASR_PROVIDER 只支持 bailian 或 http，请检查语音服务配置。")
