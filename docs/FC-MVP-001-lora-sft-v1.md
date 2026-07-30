# FC-MVP-001 local LoRA SFT v1

## Outcome

The first parameter-efficient Tool Router training gate is complete. A BF16
LoRA adapter was trained entirely locally on the frozen 160/40
train/validation v1 data, independently loaded, safely merged, and evaluated
against the unchanged 20-record eval.

The adapter materially improved routing quality, but it is not Runtime
eligible. One of two dangerous eval requests still produced a dangerous action
candidate. No Provider, Runtime, MCP, Desktop, network, Memory, Continuation,
or Lane B path was opened during training or evaluation.

## Locked experiment

```text
experiment: fc-mvp-001-lora-sft-v1
base model: Qwen/Qwen2.5-1.5B-Instruct
revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
method: BF16 LoRA
rank / alpha / dropout: 16 / 32 / 0.05
targets: q_proj, k_proj, v_proj, o_proj
sequence length: 448
epochs: 5
micro batch / accumulation / effective batch: 2 / 4 / 8
learning rate: 2e-4
scheduler / warmup: cosine / 0.1
seed: 20260729
train / validation / eval: 160 / 40 / 20
train / validation truncation: 0 / 0
```

The full contract is `configs/tool_router_lora_sft_v1.json`. Only
inference-visible instruction, available-tools, and state fields are rendered
as input. Category, split, example identifier, and gold decision are excluded
from the prompt. Loss is applied only to the canonical assistant decision JSON.

This first gate intentionally chooses the final epoch, as locked before eval.
Validation loss reached its minimum at epoch 3 and then increased; no
checkpoint was retroactively selected after examining the eval.

## Training and Adapter evidence

```text
Python: 3.12.12
PyTorch: 2.6.0+cu124
Transformers: 4.49.0
PEFT: 0.14.0
Accelerate: 1.3.0
GPU: NVIDIA GeForce RTX 4090 Laptop GPU

optimizer steps: 100
training time: 216.825720 seconds
peak allocated GPU memory: 5,217,494,016 bytes
trainable parameters: 4,358,144 (0.281521%)
final train loss: 0.002859817
final validation loss: 0.161960803
lowest validation loss: 0.134403733 (epoch 3)

adapter weight: 17,462,432 bytes
adapter weight SHA-256:
1c58a3d08598250cc01bd35a3367fbcc778c551782e6117f686394ede3d65659
independent Adapter directory: 17,468,332 bytes
```

The repository stores the independently loadable adapter under
`baseline/adapters/fc-mvp-001-lora-sft-v1`. The directory includes the adapter
weights, PEFT config, and model-card metadata; the pinned Base tokenizer is not
duplicated. `safe_merge` was also verified
without storing a redundant 3 GB merged model: the loaded and merged model
produced identical output for the pinned verification case, no LoRA parameter
tensors remained, and stderr was empty.

## Frozen eval comparison

| Metric | Prompt-only Base | LoRA SFT v1 | Delta |
|---|---:|---:|---:|
| JSON validity | 1.000000 | 1.000000 | 0.000000 |
| Decision semantic validity | 0.700000 | 0.800000 | +0.100000 |
| Tool accuracy | 0.200000 | 0.800000 | +0.600000 |
| Argument exact match | 0.000000 | 0.350000 | +0.350000 |
| Argument field F1 | 0.043478 | 0.375000 | +0.331522 |
| Risk Macro F1 | 0.425752 | 0.737302 | +0.311550 |
| Approval accuracy | 0.500000 | 0.950000 | +0.450000 |
| Rejection accuracy / recall | 0.700000 / 0.000000 | 0.950000 / 1.000000 | +0.250000 / +1.000000 |
| Fallback accuracy / recall | 0.800000 / 0.333333 | 0.900000 / 0.833333 | +0.100000 / +0.500000 |
| Dangerous false approvals | 1 | 0 | -1 |
| Dangerous action candidates | 2 | 1 | -1 |
| Duplicate action candidates | 2 | 0 | -2 |

All 20 outputs were valid JSON. Four outputs were still semantically
inconsistent, one false refusal remained, and the dangerous-action requirement
was not met. Therefore `safety_gate_passed=false` and
`runtime_eligible=false`.

Formal inference took `78.107495` seconds with `3,150,315,520` peak allocated
GPU bytes. An independent prior load/run took `67.271455` seconds and produced
the same 20 raw output strings. The formal prediction artifact SHA-256 is
`9f9e0e39b57e6b67d5892212b0f5fc8efbb8e5b3c0dfdfebd016d50ca97b5b35`.

## Reproduction

Create the local environment from `requirements/training.lock`. The pinned base
model must already be present under ignored `work/models`. Training and all
subsequent commands run with Hub and Transformers offline modes enabled.

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

.\work\training-env\Scripts\python.exe `
  .\scripts\train_tool_router_lora.py `
  --config .\configs\tool_router_lora_sft_v1.json `
  --model-dir .\work\models\Qwen2.5-1.5B-Instruct `
  --output-dir .\work\reproduced-lora-sft-v1 `
  --evidence-output .\work\reproduced-lora-sft-v1-training.json

.\work\training-env\Scripts\python.exe `
  .\scripts\run_tool_router_lora_inference.py `
  --config .\configs\tool_router_lora_sft_v1.json `
  --model-dir .\work\models\Qwen2.5-1.5B-Instruct `
  --adapter-dir .\work\reproduced-lora-sft-v1\adapter `
  --training-evidence .\work\reproduced-lora-sft-v1-training.json `
  --output .\work\reproduced-lora-sft-v1-predictions.json

python -I .\scripts\score_tool_router_lora_predictions.py `
  --config .\configs\tool_router_lora_sft_v1.json `
  --training-evidence .\work\reproduced-lora-sft-v1-training.json `
  --predictions .\work\reproduced-lora-sft-v1-predictions.json `
  --base-report .\baseline\tool-router-qwen2.5-1.5b-instruct-report.json `
  --output .\work\reproduced-lora-sft-v1-report.json

.\work\training-env\Scripts\python.exe `
  .\scripts\verify_tool_router_lora_adapter.py `
  --config .\configs\tool_router_lora_sft_v1.json `
  --model-dir .\work\models\Qwen2.5-1.5B-Instruct `
  --adapter-dir .\work\reproduced-lora-sft-v1\adapter `
  --training-evidence .\work\reproduced-lora-sft-v1-training.json `
  --output .\work\reproduced-lora-sft-v1-load-merge.json
```

PEFT's official LoRA contract documents rank, alpha, target modules, and
`save_pretrained`; the saved adapter is loaded with `PeftModel.from_pretrained`
and verified with `merge_and_unload(safe_merge=True)`.

## Limitations and next gate

- This is one LoRA configuration, not a LoRA-versus-QLoRA or rank ablation.
- The 20-record eval is useful as a frozen regression gate but too small for a
  broad deployment claim.
- The train/validation loss divergence after epoch 3 indicates overfitting.
- One dangerous action candidate and four semantic inconsistencies remain.
- No model candidate may be sent to Runtime from this adapter.

The next gate is a training-only safety repair: classify the frozen badcases,
add reviewed hard negatives to new train/validation data without changing or
copying eval answers, lock one v2 config, and require zero dangerous action
candidates and zero dangerous false approvals before any Runtime review.
