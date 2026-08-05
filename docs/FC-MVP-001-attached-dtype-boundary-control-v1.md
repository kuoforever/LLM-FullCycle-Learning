# FC-MVP-001 Attached Dtype Boundary Control v1 Evidence

Status: completed locally on 2026-08-05; remediation not yet tested; Runtime
ineligible.

## Result

The frozen experiment classifies the observed behavior as
`deterministic_same_values_rmsnorm_dtype_replay_reproduces_actual_boundary_drift`.
Four fresh BF16/FP32 attached runs reproduce the registered layer-0
`input_layernorm` output boundary at the exact frozen target forward. After
all attached models are unloaded, four fresh standalone `Qwen2RMSNorm`
executions replay the same checkpoint input and weight values under the two
locked dtypes. Each standalone output is exactly equal to its same-dtype actual
module output.

This establishes the narrow result that the same checkpoint values under the
locked BF16/FP32 `Qwen2RMSNorm` arithmetic are sufficient to reproduce this
local registered boundary. Unequal KV/history is not required to reproduce
this local output difference because the standalone control has no cache and
RMSNorm runs before attention/cache consumption.

It does not identify a unique `pow`, `mean`, `rsqrt`, cast, multiply, or CUDA
kernel root cause. It also does not establish independent downstream
propagation, an LM-head or token cause, a pristine-FP32 checkpoint comparison,
full-eval improvement, remediation, artifact promotion, or Runtime
eligibility.

The canonical record is
[`baseline/fc-mvp-001-attached-dtype-boundary-control-v1.json`](../baseline/fc-mvp-001-attached-dtype-boundary-control-v1.json),
is `142,760` bytes, and has SHA-256
`fdf4ab44b1b60853f0d5de9f231ce77557152b47c9ce52156c31c9bbca484bc7`.

## Frozen source and target

The immediate source gate is
[`FC-MVP-001-attached-dtype-numerics-v1`](../baseline/fc-mvp-001-attached-dtype-numerics-v1.json),
locked by SHA-256
`de5b048a5d254f61ab3bef1ff23f1484b07808c86a1679bc0de4ee58e8c8d7c5`.
The checkpoint, Adapter, prompt, eval, attached execution form, high-level SDPA
dispatch, greedy generation, disabled autocast/TF32, and target identity remain
unchanged.

The target is `eval-001` generation step and cached causal call `45`. Its input
is generated-token index `44`, token ID `788`, with past length and cache
position `383`; each actual run executes 48 causal forward calls. The BF16 path
predicts token `1866` (`true`) and the FP32 path predicts token `3849` (`false`),
matching the frozen attached references.

The control reads two common BF16 checkpoint sources and canonicalizes them to
FP32 for exact identity checks:

| Source | Shape | Canonical FP32 SHA-256 |
|---|---:|---|
| `model.embed_tokens.weight[788]` | `[1, 1, 1536]` | `8ebfe9e9c6947d1295014958f313cd761f84c7bbaa23312d27aa7cdca0b01d3c` |
| `model.layers.0.input_layernorm.weight` | `[1536]` | `32a0635b2116dad20776d4f270eb93090bc95233c3fa10084d4a59be1ce7839a` |

No source checkpoint or Adapter values changed. The Adapter remains attached
and FP32 on both actual paths; only the base load/inference dtype differs.

## Actual attached ABBA reproduction

The actual phase uses four fresh model-load lifecycles in fixed ABBA order:

| Order | Run | Path | Target token | Elapsed (s) | Peak allocated bytes | After release bytes |
|---:|---|---|---:|---:|---:|---:|
| 0 | `bf16-attached-boundary-r1` | BF16 attached | 1866 | `7.3087088000029325` | `3,186,210,816` | `8,519,680` |
| 1 | `fp32-attached-boundary-r1` | FP32 attached | 3849 | `6.931469400005881` | `6,285,152,256` | `8,519,680` |
| 2 | `fp32-attached-boundary-r2` | FP32 attached | 3849 | `6.9172180000459775` | `6,285,152,256` | `8,519,680` |
| 3 | `bf16-attached-boundary-r2` | BF16 attached | 1866 | `6.7880022000172175` | `3,186,210,816` | `8,519,680` |

Each run reproduces its frozen 48-token output, decoded output,
processed-score trace, raw-logit trace, target vector, precision audit, and
target-forward identity. Each dtype's two runs are exact across path and
capture records.

Four target-scoped summaries are captured per actual run:

1. `model.embed_tokens` output;
2. layer-0 `input_layernorm` input;
3. layer-0 `input_layernorm.weight` parameter; and
4. layer-0 `input_layernorm` output.

The actual phase therefore contributes 16 summary-only capture records and
four cross-dtype comparisons. The representative comparison manifest is exact
across the repeat pair.

## Standalone same-values RMSNorm control

Exactly one control is registered:
`layer0_input_rmsnorm_same_checkpoint_values_dtype_replay`. It has two dtype
arms, each repeated twice in ABBA order after every actual attached model has
been released:

| Order | Run | Input / weight / output dtype | Fresh standalone module | Cache |
|---:|---|---|---:|---:|
| 0 | `bf16-rmsnorm-control-r1` | BF16 | true | absent |
| 1 | `fp32-rmsnorm-control-r1` | FP32 | true | absent |
| 2 | `fp32-rmsnorm-control-r2` | FP32 | true | absent |
| 3 | `bf16-rmsnorm-control-r2` | BF16 | true | absent |

Every execution creates a fresh
`transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm` with hidden size
`1536`, epsilon `1e-6`, eval/inference mode, and the same checkpoint source
values. The locked `Qwen2RMSNorm.forward` source SHA-256 is
`7d352cd525210579aabf6191da9bfc1b1086878c303fb1ea8b8ea21d0e081342`.
Autocast, TF32, cache arguments, output injection, serialized tensor payloads,
and a module tensor sidecar are absent.

The control phase contributes 12 summary-only capture records and three
cross-dtype comparisons. Across both repeats, its input, weight, and output
records and comparison manifest are exact.

## Exact comparison and actual/control linkage

All comparisons use contiguous canonical FP32 values with signed-zero
normalization, finite-only validation, no tolerance, fixed flatten order, and
stdlib `math.fsum` float64 reductions.

| Value | BF16 canonical SHA-256 | FP32 canonical SHA-256 | Different / total | Max / mean / RMS absolute delta |
|---|---|---|---:|---|
| Actual embedding output | `8ebfe9e9c6947d1295014958f313cd761f84c7bbaa23312d27aa7cdca0b01d3c` | same | `0 / 1,536` | `0 / 0 / 0` |
| Actual RMSNorm input | `8ebfe9e9c6947d1295014958f313cd761f84c7bbaa23312d27aa7cdca0b01d3c` | same | `0 / 1,536` | `0 / 0 / 0` |
| Actual RMSNorm weight | `32a0635b2116dad20776d4f270eb93090bc95233c3fa10084d4a59be1ce7839a` | same | `0 / 1,536` | `0 / 0 / 0` |
| Actual RMSNorm output | `fcf241d93faf88fa991d10e987d879b33ce01ab94426e73dfab62048bfafa897` | `b37c6dc89813c2bc0977d130ef0a1befdfffeeb77474f7682be8b267d19cb499` | `1,536 / 1,536` | `0.012537479400634766 / 0.0005440729593146898 / 0.0009270972900508952` |
| Control RMSNorm input | `8ebfe9e9c6947d1295014958f313cd761f84c7bbaa23312d27aa7cdca0b01d3c` | same | `0 / 1,536` | `0 / 0 / 0` |
| Control RMSNorm weight | `32a0635b2116dad20776d4f270eb93090bc95233c3fa10084d4a59be1ce7839a` | same | `0 / 1,536` | `0 / 0 / 0` |
| Control RMSNorm output | `fcf241d93faf88fa991d10e987d879b33ce01ab94426e73dfab62048bfafa897` | `b37c6dc89813c2bc0977d130ef0a1befdfffeeb77474f7682be8b267d19cb499` | `1,536 / 1,536` | `0.012537479400634766 / 0.0005440729593146898 / 0.0009270972900508952` |

For each dtype and repeat, control input equals the common source and actual
input, control weight equals the common source and actual weight, and control
output equals the actual registered output. The actual four-comparison
manifest SHA-256 is
`029be62e5d8a342761c979f53378fc4253877846f6f6b5ef1214d5fde0abbc27`;
the control three-comparison manifest SHA-256 is
`bc225d217885bda91670e4a0b71b8a1d88c1ef55585527bd7aa2d18f32ad1ebf`.

The eight executions emit 28 summary-only capture records: 16 actual and 12
control. The artifact additionally locks two checkpoint-source summary
records. No raw module tensor payload or sidecar exists. The JSON-only
validator can independently close digests, frozen links, ordering, repeat
identity, comparison manifests, gates, and summary algebra, but it cannot
recompute the full captured tensors from digests and summaries alone.

## Causal scope and gate state

`boundary_control_gate.passed=true` and the protocol is complete. The matched
control shows that locked same-values BF16/FP32 `Qwen2RMSNorm` arithmetic is
sufficient for this local registered output boundary. It does not show which
internal operation or kernel is uniquely responsible, nor does it show that
the RMSNorm delta independently causes later registered, LM-head, or token
differences.

No new remediation is tested, so `remediation_gate.passed=false` and
`runtime_eligible=false`. The completed boundary-control gate therefore locks
`constraints.full_eval_run=false`. No full eval, new data, training,
eval-answer tuning, merge, merged-weight save, artifact promotion,
Runtime/Provider/MCP/Desktop
integration, actual-model intervention, second control, or module tensor
sidecar occurred.

## Resources and validation

The probe ran under Python 3.12.12, PyTorch 2.6.0+cu124, Transformers 4.49.0,
PEFT 0.14.0, and Accelerate 1.3.0 on one NVIDIA GeForce RTX 4090 Laptop GPU.
Total elapsed time was `32.9616448999732` seconds and peak allocated GPU memory
was `6,285,152,256` bytes. Every actual and standalone-control lifecycle
released below the locked `16,777,216`-byte ceiling.

The unified offline gate passes 249 tests with `valid=true` on CPython 3.11.15,
3.12.12, and 3.13.7, and reports `source_files_audited=27`. Ruff, mypy 2.3.0,
and py_compile also pass.

## Single next objective

`FC-MVP-001-fp32-attached-remediation-eval-v1` is eligible to start. It must
pre-register one resource-bounded full frozen evaluation of the FP32 attached
Adapter with decision compilation fixed, compare it against the frozen BF16
attached v2 metrics, and keep merge, artifact promotion, and Runtime
integration prohibited until safety, regression, and resource gates pass.
For this next gate, `locked_next_action.constraints.full_eval_run=true`: the
single pre-registered full frozen evaluation is required and allowed, while the
completed boundary-control gate itself correctly records that no full eval ran.

Acceptance requires:

- this matched boundary-control evidence to remain frozen;
- exactly one FP32 attached candidate;
- the unchanged 20-case eval and unchanged decision compiler;
- the frozen BF16 attached reference; and
- an explicit resource and safety comparison.

No data, training, eval-answer tuning, execution-form change, Adapter runtime
dtype change, source checkpoint change, backend/decoding change, merge,
artifact promotion, or Runtime integration is allowed.

## Exact reproduction commands

```powershell
.\work\training-env\Scripts\python.exe `
  .\scripts\probe_tool_router_attached_dtype_boundary_control.py `
  --config .\configs\tool_router_lora_sft_v2.json `
  --model-dir .\work\models\Qwen2.5-1.5B-Instruct `
  --adapter-dir .\baseline\adapters\fc-mvp-001-lora-sft-v2 `
  --numerics-evidence .\baseline\fc-mvp-001-attached-dtype-numerics-v1.json `
  --output .\work\test-fixtures\fc-mvp-001-attached-dtype-boundary-control-v1.json

.\work\training-env\Scripts\python.exe -I .\scripts\validate_offline.py
```

The probe implementation is
[`scripts/probe_tool_router_attached_dtype_boundary_control.py`](../scripts/probe_tool_router_attached_dtype_boundary_control.py).
The classification helper and strict JSON validator are
[`tool_router_attached_dtype_boundary_control.py`](../src/fullcycle_bridge/tool_router_attached_dtype_boundary_control.py)
and
[`tool_router_attached_dtype_boundary_control_evidence.py`](../src/fullcycle_bridge/tool_router_attached_dtype_boundary_control_evidence.py).
