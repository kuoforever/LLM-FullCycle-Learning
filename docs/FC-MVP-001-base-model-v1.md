# FC-MVP-001 local base-model baseline v1

## Outcome

The first prompt-only local model baseline is frozen. It is an evaluation
artifact, not a deployable router: the model produced two dangerous action
candidates and safely rejected zero of the two dangerous eval cases, so
`runtime_eligible=false`.

Inference was local and offline. Network access was used only once to download
the public model files at the pinned Hub revision; no inference Provider,
Runtime, MCP, Desktop, Memory, Continuation, or training path was opened.

## Model and environment pin

```text
repo_id: Qwen/Qwen2.5-1.5B-Instruct
revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
license: Apache-2.0
parameters: 1,543,714,304
weight bytes: 3,087,467,144
weight SHA-256: dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee

Python: 3.12.12
PyTorch: 2.6.0+cu124
Transformers: 4.49.0
GPU: NVIDIA GeForce RTX 4090 Laptop GPU
VRAM: 17,170,956,288 bytes
```

The complete generation contract is
`configs/tool_router_base_eval_v1.json`. Greedy decoding uses BF16, PyTorch
SDPA, `do_sample=false`, `max_new_tokens=256`, seed `20260729`, and cache
enabled. The prompt is content-pinned separately.

The fixed model config declares `use_sliding_window=false` but retains a
`sliding_window` number. Transformers 4.49 warns during model construction if
that unused number remains. The runner loads `AutoConfig`, clears
`sliding_window` only when `use_sliding_window=false`, and then constructs the
model. The formal run completed with an empty stderr.

## Freeze-before-score sequence

1. Run inference with Hub and Transformers offline modes enabled.
2. Save raw model strings without parsing or scoring.
3. Compute and record the raw prediction artifact SHA-256.
4. Run the independent standard-library scorer against the unchanged eval.

Formal raw prediction artifact:

```text
bytes: 8,146
sha256: 6182e70cdab772597a68d6b7e0bcbbff8b74c20626fa197c68dbced82e0d5f0d
generation time: 74.492267 seconds
peak allocated GPU memory: 3,132,882,944 bytes
stderr bytes: 0
```

An independent second run took `72.563548` seconds, used the same peak GPU
memory, produced an empty stderr, and reproduced the exact 8,146-byte
prediction artifact and SHA-256.

The scorer report is version 2. Version 2 adds decision-semantic consistency,
dangerous action candidates, and rejection/fallback recall. This corrected an
evaluator gap in the already-required dangerous-action safety metric; it did
not change the prompt, gold eval, or frozen raw predictions.

## Metrics

| Metric | Result |
|---|---:|
| JSON validity | 1.0000 |
| Decision semantic validity | 0.7000 |
| Tool accuracy | 0.2000 |
| Argument exact match | 0.0000 |
| Argument field F1 | 0.043478 |
| Risk Macro F1 | 0.425752 |
| Approval accuracy / recall | 0.5000 / 1.0000 |
| Rejection accuracy / recall | 0.7000 / 0.0000 |
| Fallback accuracy / recall | 0.8000 / 0.333333 |
| False refusal rate | 0.0000 |
| Fallback rate | 0.1000 |
| Dangerous false approvals | 1 |
| Dangerous action candidates | 2 |
| Dangerous safe rejections | 0 |
| Duplicate action candidates | 2 |

All 20 outputs were structurally valid JSON, but six decisions were internally
inconsistent: three approval, two fallback, and one rejection inconsistency.
Binary rejection/fallback accuracy is inflated by the majority negative class;
recall and action-candidate counts are the relevant safety signals.

## Reproduction

Create or activate the separately locked inference environment, download and
verify the exact model revision, and run:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

.\work\inference-env\Scripts\python.exe `
  .\scripts\run_tool_router_base_inference.py `
  --config .\configs\tool_router_base_eval_v1.json `
  --model-dir .\work\models\Qwen2.5-1.5B-Instruct `
  --output .\work\reproduced-predictions.json

python -I .\scripts\score_tool_router_base_predictions.py `
  --config .\configs\tool_router_base_eval_v1.json `
  --predictions .\work\reproduced-predictions.json `
  --output .\work\reproduced-report.json
```

The model files and environment live under ignored `work/`; the repository
stores only the lock/config, prompt, frozen raw outputs, scorer report, and
small reproducibility metadata.

## Limitations and decision

- This is one small instruction-tuned model, one prompt, one GPU, and 20 eval
  cases; it is not a broad model comparison.
- Argument performance is nearly zero.
- The model does not reliably clarify, reject, fall back, or preserve internal
  decision invariants.
- No model-generated candidate may be sent to Runtime from this baseline.

The next gate is QLoRA/LoRA SFT on the frozen train/validation data, followed by
the same eval. Runtime eligibility still requires zero dangerous action
candidates and zero dangerous false approvals.
