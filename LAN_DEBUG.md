# 安卓设备局域网联调

目标：先确认设备能访问 FastAPI，再验证普通问答、流式回答和语音上传。现有 RAG、FAQ、数据库、百炼配置和 Streamlit 页面无需为此调整。Streamlit 可以继续只供电脑本机管理。

## 1. 网络和服务

- 电脑和安卓设备连接同一个允许设备互访的网络。电脑接网线、设备接 Wi-Fi 也可以，只要网络允许互访。
- 避免访客 Wi-Fi、AP/客户端隔离；校园网即使名称相同，也可能禁止设备互访。
- 电脑需要继续联网调用百炼。调试时保持开机、服务运行，避免睡眠。
- 使用 `ipconfig` 查实际 WLAN 或有线网卡的 IPv4；不要选 VMware、Hyper-V、VPN 的地址。
- 下面的 `192.168.1.20` 和 `192.168.1.0/24` 是例子，必须替换成实际地址和子网。

在项目目录启动：

```powershell
.\start_api.ps1 -ListenAddress 192.168.1.20
```

这会只监听所选网卡的 IPv4，端口默认 8001，单工作进程。默认不传 `-ListenAddress` 时仍只监听 `127.0.0.1`。若同一监听地址的端口已有服务，应先在原窗口按 Ctrl+C 关闭，或显式指定其他端口：

```powershell
.\start_api.ps1 -ListenAddress 192.168.1.20 -Port 8002
```

绑定到具体局域网 IPv4 的服务不提供 `127.0.0.1` 入口；电脑和设备均使用该 IPv4。已有独立的本机服务可以保留，但两者使用同一数据目录，不要同时进行大量写入。

Uvicorn 也支持 `0.0.0.0` 监听所有 IPv4 网卡；本项目的局域网联调优先指定具体地址，避免同时开放到 VPN 等其他网卡。`0.0.0.0` 是监听地址，不能填写为 APP 的服务地址。电脑 IP 改变后需要重新启动并更新 APP 地址；长期调试可在路由器中做 DHCP 地址保留。

## 2. Windows 防火墙

服务启动成功不等于其他设备能连接。先检查当前网络类型及已有规则，不要关闭整个防火墙。需要新规则时，在管理员 PowerShell 中添加仅限实际网卡、本机 IPv4、实际子网和 API 端口的规则。以下示例适用于 /24 子网：

```powershell
New-NetFirewallRule -Name 'RAG-LAN-API-8001' -DisplayName 'RAG LAN API 8001' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8001 -LocalAddress 192.168.1.20 -RemoteAddress 192.168.1.0/24 -InterfaceAlias 'WLAN' -Profile Private,Public
```

网卡名称、地址、子网、端口均需与实际环境一致。规则适用于 Private/Public，但只接受指定网卡、地址和子网的连接，不改变当前网络类型。若已有同名规则，先检查规则再调整，避免重复添加。无需配置路由器公网端口转发。

调试结束后如需撤销此规则：

```powershell
Remove-NetFirewallRule -Name 'RAG-LAN-API-8001'
```

目前接口没有鉴权，包含文档上传和 FAQ 管理功能，只应在可信的调试网络内开放。

## 3. 从设备逐步验证

将下方地址替换成启动时的 IPv4 和端口。

| 顺序 | 检查 | 成功标准 |
| --- | --- | --- |
| 1 | 电脑浏览器打开 `http://192.168.1.20:8001/health` | `{"status":"ok"}` |
| 2 | 安卓设备浏览器打开相同 `/health` 地址 | 相同 JSON，证明设备到接口的基本 HTTP 连通 |
| 3 | 设备浏览器打开 `/docs` | 显示交互文档；若文档页面加载失败，仍以 `/health` 判断基本连通，因为文档 UI 还依赖外部静态资源 |
| 4 | 调用 `GET /knowledge/status`、`GET /faqs` | 返回原有知识段数和 FAQ |
| 5 | 调用 `POST /chat` | FAQ 标准问题返回保存的答案；普通问题可正常回答 |
| 6 | APP 调用 `POST /chat/stream` | 回答逐段显示，能识别 `done` 和 `error` |
| 7 | APP 上传短 WAV 至 `/audio/transcriptions` | 得到文字，填回输入框供编辑，再调用聊天接口 |

`/health` 不请求模型、不初始化知识库。它通过后，模型鉴权、百炼外网连接仍需在后续步骤单独验证。浏览器地址栏不能直接测试 POST；可使用 `/docs` 的 Try it out、接口工具或 APP。文档 UI 未必逐段展示 SSE，以 APP 或流式客户端的表现为准。

如果电脑访问成功、设备失败：先检查是否用了正确的网卡地址，再检查设备所在网络、Windows 防火墙、路由器隔离、VPN/代理及电脑是否休眠。能打开 `/health` 但 APP 失败时，优先检查 APP 的 HTTP 配置和权限。

## 4. 安卓 APP 最小准备

- 将服务地址设置为可修改的调试配置，例如 `http://192.168.1.20:8001`。无需把百炼 API Key 放进 APP。
- 在 manifest 声明 `android.permission.INTERNET`；录音还需 `android.permission.RECORD_AUDIO` 并在运行时请求麦克风授权。网络请求放在后台执行。
- 面向 Android 9 / targetSDK 28 及以上的 APP 默认限制明文 HTTP。在 debug 构建中为测试主机放行 HTTP；若已有 `networkSecurityConfig`，应修改对应配置，不能只依赖 `usesCleartextTraffic`。正式公网版本改用 HTTPS。
- Android 17 且 targetSDK ≥ 37 时，还需声明并运行时申请 `ACCESS_LOCAL_NETWORK`。targetSDK ≤ 36 不应直接添加或申请该权限；Android 16 的局域网限制是可选测试功能。联调时先确认设备系统版本和 APP 的 targetSDK，再决定是否需要这一项。
- 流式聊天是带 JSON 请求体的 **POST SSE**，不要使用只支持 GET 的 EventSource。按 UTF-8 解码、空行划分 SSE 事件，再解析 JSON；一次网络读取不等于一个完整事件。事件为 `sources`、`token`、`done`、`error`，HTTP 200 也可能包含 `error` 事件。
- 客户端需设置适合生成过程的超时，避免整体响应缓冲；支持取消请求。聊天历史由 APP 保存，按 API.md 传 `role/content`。
- 录音上传字段为 `file`。优先 16 kHz、单声道 PCM WAV；安卓常见 AAC/M4A 不能直接当作 WAV 上传，需选择适配器支持的格式或先转换。录音结束→转写→填入输入框→用户编辑/发送。
- 原生 APP 通常不受浏览器 CORS 约束，无需为它添加全开放 CORS。若 APP 实际使用 WebView/H5，则另行检查其来源和安全策略。

若用手机浏览器访问 Streamlit 的局域网 HTTP 页面，录音还会受到浏览器安全上下文限制；网页能够聊天不代表麦克风可用。原生 APP 录音后上传到 FastAPI 不依赖网页麦克风功能。

## 参考

- [Android 网络连接与权限](https://developer.android.com/develop/connectivity/network-ops/connecting)
- [Android Network Security Configuration](https://developer.android.com/privacy-and-security/security-config)
- [Android 局域网权限与目标版本](https://developer.android.com/privacy-and-security/local-network-permission)
- [Windows 防火墙规则参数](https://learn.microsoft.com/en-us/powershell/module/netsecurity/new-netfirewallrule)
- [Uvicorn 监听地址](https://www.uvicorn.org/settings/)
- 请求与响应细节见 [API.md](API.md)。
