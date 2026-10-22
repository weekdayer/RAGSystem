# FastAPI 接口

启动：

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001
```

默认地址：`http://127.0.0.1:8001`。交互文档：<http://127.0.0.1:8001/docs>。当前接口面向本机单人使用，所有调用共享本地知识库；模型 Key 由服务端 `.env` 提供，请求不接收供应商 Key。

局域网联调可使用 `./start_api.ps1 -ListenAddress <电脑的局域网IPv4>`，默认端口仍为 8001。设备访问该 IPv4，不使用 `127.0.0.1` 或 `0.0.0.0`。网络、防火墙、Android 调试配置和验收步骤见 [LAN_DEBUG.md](LAN_DEBUG.md)。

## 接口一览

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health` | 存活检查，不请求模型、不初始化知识库 |
| GET | `/knowledge/status` | 当前知识库文本段数 |
| POST | `/documents` | 上传 TXT、Markdown 或文字型 PDF |
| POST | `/chat` | 一次性返回回答与相关资料 |
| POST | `/chat/stream` | SSE 增量回答 |
| POST | `/audio/transcriptions` | 音频转文字，底层 ASR 可替换 |
| GET | `/faqs` | FAQ 列表和条数 |
| POST | `/faqs` | 添加一条 FAQ，重复内容幂等 |
| DELETE | `/faqs/{faq_id}` | 删除指定 FAQ，不存在返回 404 |
| POST | `/faqs/import` | 上传 JSON 批量导入 FAQ |
| GET | `/faqs/settings` | 当前共享 FAQ 匹配设置 |
| PUT | `/faqs/settings` | 保存共享 FAQ 匹配设置，后续查询立即采用 |
| POST | `/faqs/match` | 测试 FAQ 匹配与候选分数，不生成模型回答 |

## 聊天

普通与流式聊天使用相同请求格式：

```json
{
  "question": "那报名截止时间呢？",
  "history": [
    {"role": "user", "content": "读书会什么时候举行？"},
    {"role": "assistant", "content": "每周六下午 14:00 到 16:00。"}
  ]
}
```

`question` 去除首尾空白后为 1～4000 字符。`history` 可省略；最多接收 100 条，每条角色为 `user` 或 `assistant`，每条文字为 1～4000 字符。核心只采用最近 12 条有效消息，总历史上限 18,000 字符。客户端应只传 `role` / `content`，不要把页面保存的 `sources` 字段带入历史。

系统身份和回答规则由服务端 `prompts.py` 统一预置，普通与流式接口共用，客户端无需拼接系统提示词，也不接受历史中传入 `system` 角色。助手应用身份为“本地知识智能助手”；明确询问底层模型时按服务端实际配置说明。

普通响应：

```json
{
  "answer": "报名截止时间是每周五 18:00 前。[1]",
  "mode": "knowledge",
  "sources": [
    {
      "content": "报名方式：参加者应在每周五 18:00 前填写报名表。",
      "metadata": {"source": "knowledge.md", "relevance_score": 0.72}
    }
  ]
}
```

`mode=faq` 表示直接返回匹配到的 FAQ 标准答案；`mode=knowledge` 表示有相关本地文档，`mode=model` 表示直接由模型回答，此时 `sources=[]`。FAQ 来源包含 `kind=faq`、`faq_id`、`faq_question`、`match_method`（`exact` 或 `semantic`）与 `relevance_score`；文档来源可能包含 PDF 的 `page`、分块标识等字段。

## SSE 流式回答

`POST /chat/stream` 返回 `Content-Type: text/event-stream`。每条事件以空行分隔，`data` 是完整 JSON，网络分块与事件边界不一定一致。

事件顺序：

```text
event: sources
data: {"type":"sources","sources":[],"mode":"model"}

event: token
data: {"type":"token","content":"你好"}

event: token
data: {"type":"token","content":"，有什么可以帮你？"}

event: done
data: {"type":"done","answer":"你好，有什么可以帮你？"}

```

`sources` 在回答开始前发送；普通模型回答的 `token.content` 是模型实际产生的增量文本，粒度不保证为单字。FAQ 命中使用相同协议，返回一个包含完整标准答案的 `token` 事件，不额外调用模型或模拟逐字生成。`done.answer` 是完整答案。建议客户端收到 `done` 后再把完整回答加入历史。

流开始后的错误以事件返回，之后不再发送 `done`：

```text
event: error
data: {"type":"error","message":"模型服务暂时不可用，请稍后重试。","status_code":503}

```

此时 HTTP 响应头可能已是 200，必须检查事件类型。请求参数或初始化失败则直接返回 HTTP 错误。断线时服务关闭模型迭代器；当前不支持断点续传，需重新提交问题。

Python 调用示例：

```python
import json
import httpx

with httpx.stream(
    "POST",
    "http://127.0.0.1:8001/chat/stream",
    json={"question": "什么是机器学习？", "history": []},
    timeout=120,
) as response:
    response.raise_for_status()
    for line in response.iter_lines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        if event["type"] == "token":
            print(event["content"], end="", flush=True)
        elif event["type"] == "error":
            print("\n请求失败：", event["message"])
```

## FAQ Demo

`POST /faqs` 接收 JSON，例如：

```json
{
  "question": "如何联系人工客服？",
  "answer": "进入帮助与客服页面，选择联系人工客服。",
  "aliases": ["转人工", "我要找人工客服"]
}
```

成功返回包含 `id`、`question`、`answer`、`aliases`、`updated_at` 的条目。相同问题与内容再次添加返回原条目；内容冲突返回 400。`GET /faqs` 返回 `{"items": [...], "count": 6}`；`DELETE /faqs/{faq_id}` 成功返回 `{"deleted": true}`，不存在返回 404。Demo 暂不提供编辑接口，需要先删除再添加。

`POST /faqs/import` 接收 `multipart/form-data` 的 `file` 字段，文件扩展名 `.json`，内容为上述条目的 JSON 数组；可以直接上传 `examples/faq.json`。成功返回 `{"added": 6, "skipped": 0, "count": 6}`，重复导入跳过相同内容。文件最多 1 MiB、一次最多 500 条；问题最多 4000 字符，答案最多 12000 字符，别名最多 20 条、每条最多 4000 字符。格式或问法冲突时整次导入失败，不保留半批数据。

问答入口仍是 `/chat` 和 `/chat/stream`，不需要另调 FAQ 检索接口：当前问题优先精确匹配标准问题或别名，其次用向量与 BM25 中文关键词混合检索；未命中继续文档 RAG 和模型聊天。混排第一名的原始余弦分低于 0.70（70 分）时不会返回 FAQ，关键词分不能绕过门槛；候选领先差不足时也继续 RAG。

`GET /faqs/settings` 返回 `{"threshold": 0.7, "min_margin": 0.08}`。`PUT /faqs/settings` 接收 `{"threshold": 0.7}`，可选 `min_margin`，省略时保留当前领先差。字段必须为有限数值，范围 0～1，不接受百分数 70 或供应商密钥。保存至 SQLite 后，网页和接口的下一次查询即采用；`.env` 中的 `FAQ_MATCH_THRESHOLD`（默认 0.70）与 `FAQ_MIN_MARGIN`（默认 0.08）仅为尚未保存配置时的初始值。

`POST /faqs/match` 接收 `{"question": "退款在哪里？"}`，返回 `question`、`matched`、`reason`、`threshold`、`min_margin`、`match` 和 `candidates`。命中条目 `match` 含 `id/question/answer/match_method`，未命中为 null；候选含 `id/question/semantic_score/bm25_score/hybrid_score`。`reason` 为 `exact/hybrid/empty/below_threshold/ambiguous/no_candidates`；语义与混合分可能为负（有限数，范围 -1～1），BM25 是非负原始分值，不是概率。非精确来源仍使用 `match_method=semantic` 兼容旧客户端，另外附带 `retrieval_method=hybrid`、`bm25_score` 和 `hybrid_score`。

FAQ 正式问答与匹配设置保存于当前 Chroma 目录下的 `faq.sqlite3`；网页和接口共用，语义索引按向量配置自动补齐。FAQ 管理页位于 <http://127.0.0.1:8501/faq>。

## 上传文档

`POST /documents`，使用 `multipart/form-data`，字段名为 `file`。单文件最多 20 MiB；支持 `.txt`、`.md`、`.pdf`，文字型 PDF 才能提取内容。

```python
import httpx

with open("examples/knowledge.md", "rb") as file:
    response = httpx.post(
        "http://127.0.0.1:8001/documents",
        files={"file": ("knowledge.md", file, "text/markdown")},
        timeout=120,
    )
    response.raise_for_status()
    print(response.json())
```

响应：`{"filename":"knowledge.md","added_chunks":1,"count":1}`。重复文件的 `added_chunks` 为 0。

## 音频转文字

`POST /audio/transcriptions`，使用 `multipart/form-data` 的 `file` 字段：

```python
import httpx

with open("recording.wav", "rb") as file:
    response = httpx.post(
        "http://127.0.0.1:8001/audio/transcriptions",
        files={"file": ("recording.wav", file, "audio/wav")},
        timeout=120,
    )
    response.raise_for_status()
    text = response.json()["text"]
```

响应：`{"text":"识别出的文字"}`。确认文字后，再将其作为 `question` 发送到聊天接口。

默认百炼 ASR 复用聊天的 Key 和地址，也可设置独立 `ASR_API_KEY`、`ASR_BASE_URL`、`ASR_MODEL`。推荐 WAV；音频原文件上限 10 MiB，百炼还要求 Base64 编码后不超过 10 MiB。WAV 时长上限 5 分钟，其他格式的时长由上游校验。原始音频不加入知识库。

## 接入实验室 ASR

实验室服务只需实现以下协议：

- 请求：`POST` 到 `ASR_HTTP_URL`，`multipart/form-data`，二进制音频字段 `file`。
- 可选认证：`Authorization: Bearer <ASR_HTTP_API_KEY>`。
- 成功响应：HTTP 200，JSON `{"text":"识别结果"}`；`text` 必须为非空字符串。
- 服务地址应直接指向实验室识别接口，不要指回本项目的 `/audio/transcriptions`，也不要依赖重定向。

配置示例：

```dotenv
ASR_PROVIDER=http
ASR_HTTP_URL=http://127.0.0.1:9000/transcribe
```

重启网页和接口服务后即使用实验室模型，无需改聊天、检索或页面逻辑。也可直接实现 `speech.SpeechToText` 协议，在 `create_transcriber` 注册新实现。

## 错误与状态

非流式失败返回 `{"detail":"可读提示"}`，不包含供应商响应正文、Key 或请求头。

| 状态 | 含义 |
| --- | --- |
| 400 | 输入、文件或识别内容无效 |
| 413 | 上传文件超过接口上限 |
| 502 | 上游服务拒绝请求或语音服务失败 |
| 503 | 服务未配置、连接失败、超时、限流或上游暂不可用 |
| 500 | 其他处理错误 |

健康检查仅表明进程可用，不代表模型鉴权或网络已经通过。API 检索默认门槛可用 `RAG_RELEVANCE_THRESHOLD` 设置；网页侧栏的修改只作用于当前网页会话。
