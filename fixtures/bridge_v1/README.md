# Bridge v1 compatibility fixtures

`valid/runtime-manifest.json` 由固定 Runtime commit
`8ace897f746a4aa3dd3f8b10af392ea9ba81941d` 的离线 producer command
生成。`valid/minimal-run-export.json` 是最小完整 consumer 样例。

`fixture-metadata.json` 固定 producer commit、PR、全部 schema/contract 版本、
consumer schema 版本和 canonical manifest digest。

`invalid/` 保存以下 fail-closed 样例：

| Fixture | 预期错误 |
|---|---|
| `unknown-version.json` | `UNSUPPORTED_VERSION` |
| `digest-mismatch.json` | `MANIFEST_DIGEST_MISMATCH` |
| `wrong-data-class.json` | `INVALID_DATA_CLASS` |
| `wrong-training-use.json` | `INVALID_TRAINING_USE` |
| `unexpected-field.json` | `UNKNOWN_FIELD` |
| `malformed.json` | `MALFORMED_JSON` |
| `incomplete-event.json` | `MISSING_FIELD` |
| `rich-content.json` | `FORBIDDEN_RICH_FIELD` |

Oversized input、checkpoint 和 event 在测试中按真实边界常量构造，避免把
24 MiB 垃圾文件提交到仓库。
