# FC-BRIDGE-001：Runtime Lane A 离线消费契约

English: [en/FC-BRIDGE-001.md](en/FC-BRIDGE-001.md)

## 范围

本模块只消费 `guarded-desktop-agent` 自动导出的 Lane A 安全证据：

```text
runtime manifest v1 + redacted run-export v1
    -> bounded JSON read
    -> exact schema/version validation
    -> canonical manifest SHA-256 binding
    -> redacted reliability evidence
```

固定 producer：

```text
runtime_git_commit=8ace897f746a4aa3dd3f8b10af392ea9ba81941d
runtime_pull_request=219
agent_contract_version=0.1.0
driver_contract_version=1.0.0
fullcycle_manifest_version=1
fullcycle_run_export_version=1
trace_version=1
checkpoint_version=1
plan_contract_version=1
consumer_schema_version=1.0.0
```

consumer 使用 Python 标准库，无运行时第三方依赖。校验过程只读取用户明确传入的两个普通文件，不导入或启动 Provider、MCP、Desktop、网络、Approval、Memory 或 Continuation。

## 接受条件

Manifest 必须：

- 顶层字段集合精确匹配 v1；
- 六个公开 contract/schema 版本精确匹配；
- `automatic_export` 精确包含六项声明，且全部为 `false`；
- 每个 tool 精确包含 producer v1 的十二个字段；
- tool name 唯一，枚举、布尔、字符串列表和 JSON input schema 类型合法。

Run export 必须：

- `fullcycle_run_export_version=1`；
- `manifest_digest` 等于对已验证 manifest 进行 UTF-8、sorted-key、compact canonical JSON 编码后的 SHA-256；
- `data_class=redacted_runtime_evidence`；
- `training_use=reliability_and_verifier_only`；
- 顶层、checkpoint、budgets、metrics 和按 kind 区分的 event 字段严格匹配；
- run/checkpoint/event 的 `run_id` 一致；
- `event_count`、从 1 开始的连续 sequence 和实际事件数一致；
- 第一个事件是与 checkpoint `task_length` 一致的完整 `user_task`；
- phase、recovery、policy、tool status 和 dispatch 使用 producer v1 的已知枚举；
- budget used 值不超过对应上限。

大小边界：

| 对象 | 最大值 |
|---|---:|
| 每个输入 JSON 文件 | 24 MiB |
| canonical checkpoint | 64 KiB |
| 每个 canonical event | 1 MiB |

解析同时拒绝空文件、符号链接、非 UTF-8、duplicate JSON key、`NaN`/`Infinity` 和 malformed JSON。

## Fail-closed 行为

`BridgeValidationError` 提供稳定的 `code` 和 `location`。CLI 成功退出码为 `0`；校验失败输出紧凑 JSON 到 stderr，并返回 `2`。主要错误码包括：

- `UNSUPPORTED_VERSION`
- `UNSAFE_AUTOMATIC_EXPORT`
- `MANIFEST_DIGEST_MISMATCH`
- `INVALID_DATA_CLASS`
- `INVALID_TRAINING_USE`
- `INPUT_TOO_LARGE` / `CHECKPOINT_TOO_LARGE` / `EVENT_TOO_LARGE`
- `MALFORMED_JSON` / `DUPLICATE_JSON_KEY` / `NONFINITE_NUMBER`
- `UNKNOWN_FIELD` / `MISSING_FIELD`
- `FORBIDDEN_RICH_FIELD`
- `INCOMPLETE_EVENTS` / `RUN_ID_MISMATCH`

额外富内容字段会被拒绝，包括 task/model text、tool-result text、image、memory、continuation、response 和 content。Lane A 不能用于 instruction following、GUI grounding、多模态 SFT 或行为模仿。

## 可复制离线验证

在仓库根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\validate_bridge.ps1
```

直接运行 CLI：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m fullcycle_bridge `
  --manifest .\fixtures\bridge_v1\valid\runtime-manifest.json `
  --run-export .\fixtures\bridge_v1\valid\minimal-run-export.json
```

本任务不要求访问网络，也不要求安装 package。`pyproject.toml` 记录 Python `>=3.11` 和空依赖集合。

## 限制

- consumer 验证 producer v1 的结构、安全分类和绑定关系，不验证 Runtime 执行是否业务成功。
- manifest 使用 SHA-256 内容绑定，不是带密钥的真实性签名；fixture 的 producer commit pin 由版本控制保护。
- Lane B 富多模态采集仍未实现、未批准，也不属于本模块。
- 当前本机只验证 Python 3.13.7；`requires-python` 声明为 3.11 及以上，后续可在 CI 中扩展 3.11–3.13 矩阵。
