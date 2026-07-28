# Reliability dataset v1 fixtures

本 fixture set 固定两个已通过 bridge v1 consumer 的 Lane A run export，并
映射为两行 canonical JSONL：

| Input | 覆盖信号 |
|---|---|
| `failure-denial-recovery-budget-sequence.json` | failure、policy denial、recovery、tool-call budget limit、双工具 sequence/outcomes |
| `unknown-outcome.json` | unknown dispatch/outcome、requires re-observation、单工具 sequence/outcome |

`expected-records.jsonl` 是 exact-output fixture。每个 record 都保留
`data_class=redacted_runtime_evidence` 和
`training_use=reliability_and_verifier_only`，并用 canonical run-export
SHA-256 绑定源快照。

非法输入复用 `fixtures/bridge_v1/invalid/`。Mapper 必须先通过 bridge gate，
并另外拒绝空 batch 和重复 `run_id`。测试会逐字节比较 CLI JSONL 和固定输出。
