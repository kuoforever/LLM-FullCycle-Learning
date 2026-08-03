# FC-MVP-001 local LoRA SFT v2

## Outcome

The safety-repair LoRA SFT v2 experiment is frozen. It trained entirely
locally on the passed 176/48 v2 train/validation data and was scored once
against the unchanged 20-record eval. Tool accuracy rose from v1's `0.80` to
`0.95`; dangerous action candidates fell from one to zero, so the narrow
dangerous-action safety gate passed.

The adapter remains Runtime ineligible. Three outputs contain conflicting
decision flags, three cases are false refusals, and the independently loaded
adapter and safely merged model differ by one boolean field on `eval-001`.
The merge removed all LoRA parameter tensors, but output identity failed and
is recorded rather than waived.

## Locked experiment

```text
experiment: fc-mvp-001-lora-sft-v2
base model: Qwen/Qwen2.5-1.5B-Instruct
revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
method: BF16 LoRA
rank / alpha / dropout: 16 / 32 / 0.05
targets: q_proj, k_proj, v_proj, o_proj
sequence length: 448
epochs: 3
micro batch / accumulation / effective batch: 2 / 4 / 8
learning rate: 2e-4
scheduler / warmup: cosine / 0.1
seed: 20260803
train / validation / eval: 176 / 48 / 20
train / validation truncation: 0 / 0
```

The three-epoch limit was locked before any v2 eval output was generated. It
uses the v1 validation-loss minimum as pre-existing evidence against another
five-epoch run. The canonical config digest is
`sha256:5a038ea786526f188c796a6e5eea4c4d3aa47fc66977dc4f6ff16f52999236d8`.

## Training evidence

Training completed 66 optimizer steps in `169.527236` seconds. Peak allocated
GPU memory was `5,217,494,016` bytes. The adapter has 4,358,144 trainable
parameters (`0.281521%`) and occupies 17,468,332 bytes; its 17,462,432-byte
weight file has SHA-256
`efb62471e105b8ef25641200967d447b8cc2f3ff565937bc47193fbf79f4f342`.

Validation loss decreased each epoch:

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 0.6745195180 | 0.2410457161 |
| 2 | 0.1243991707 | 0.1542072588 |
| 3 | 0.0481643728 | 0.1448025729 |

## Frozen evaluation

| Metric | LoRA v1 | LoRA v2 |
|---|---:|---:|
| JSON validity | 1.00 | 1.00 |
| Decision semantic validity | 0.80 | 0.85 |
| Tool accuracy | 0.80 | 0.95 |
| Argument exact match | 0.35 | 0.20 |
| Risk Macro F1 | 0.7373 | 0.7095 |
| Dangerous action candidates | 1 | 0 |
| Dangerous false approvals | 0 | 0 |

The remaining semantic failures are `CONFLICTING_DECISION_FLAGS` on
`eval-001`, `eval-014`, and `eval-020`. The frozen report therefore records
`safety_gate_passed=true` for the narrow dangerous-action requirements and
`runtime_eligible=false` for the overall delivery decision.

No Provider, Runtime, MCP, Desktop, network, Memory, Continuation, or Lane B
path was opened. The next gate must classify the decision inconsistencies,
false refusals, and load/merge output drift before another training change is
locked.
