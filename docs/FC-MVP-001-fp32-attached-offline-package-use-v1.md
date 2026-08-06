# FC-MVP-001 FP32 Attached Offline Package Use and Limitations v1

Status: metadata contract only. This document does not promote an artifact,
establish behavioral reproducibility, or authorize serving or Runtime use.

## Intended package identity

The package identity is the exact composition of:

- the pinned `Qwen/Qwen2.5-1.5B-Instruct` base snapshot at revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`;
- the unchanged three-file `fc-mvp-001-lora-sft-v2` Adapter;
- `prompts/tool_router_v1.txt`;
- `compile_decision` version 1 plus its direct repository dependencies;
- the locked FP32 attached generation, precision, and environment contracts;
  and
- this use-and-limitations document.

The composite manifest records logical component roles and exact file hashes.
It does not embed an absolute or caller-supplied machine path. It does record
the Adapter's historical machine-relative
`adapter_config.json.base_model_name_or_path` value, but that hint has no
authority to select a base model.

## Reconstruction and validation

1. Materialize the pinned base snapshot with
   `scripts/download_pinned_tool_router_model.py --local-dir <base-dir>`.
2. Keep the Adapter in attached factorized LoRA form. Do not merge or save
   model weights.
3. Run
   `python scripts/build_tool_router_fp32_attached_offline_package_manifest.py --check --base-model-dir <base-dir>`.
4. Treat a missing file, symbolic link, unexpected root-level regular file,
   byte-count difference, or SHA-256 difference as an ineligible local
   resolution. Do not fall back to a similarly named model or tokenizer.

The downloader and manifest bind nine base-snapshot root files. The `.cache`
directory is transport metadata and is never an identity source. A manifest
that validates without local model files proves metadata integrity only; it
does not prove that a runnable package is present on that machine.

## Required execution contract

- Load the unchanged BF16 checkpoint source values as FP32.
- Load the unchanged FP32 Adapter with PEFT in attached factorized LoRA form.
- Keep `merge=false`; do not call `merge_and_unload` or save merged weights.
- Use `torch_dtype=float32`, `attn_implementation=sdpa`, `tf32=false`, and
  `autocast=false`.
- Use deterministic greedy generation with the exact generation parameters in
  the composite manifest.
- Use the exact prompt bound by the manifest.
- Parse and validate the raw decision, then apply the required
  `compile_decision` version 1 before consuming the terminal decision fields.

The compiler is required because the frozen raw FP32 result has decision
semantic validity `0.80`, while the fixed-compiler result reaches `1.0`. The
compiler does not grant tool authority. Any later system must still validate
the schema and enforce Policy, Approval, WAL, grounding, budgets, and the sole
desktop execution boundary independently.

## Supported evidence scope

The frozen favorable evidence is limited to one pre-registered 20-case local
evaluation run with the fixed compiler. Compiled argument exact match is
`0.25`, compiled argument field F1 is `0.29787234042553196`, and the registered
safety and resource gates pass. Two prior fresh attached canary loads were
repeat-stable for one frozen example.

Those records support selecting this exact identity for a later clean-location
reproducibility gate. They do not establish cross-machine, cross-driver,
cross-library, full-evaluation, or serving reproducibility.

## Prohibited and unsupported uses

This metadata gate does not authorize:

- treating the Adapter alone as the candidate;
- substituting another base revision, tokenizer, prompt, compiler, dependency,
  precision mode, attention implementation, or generation parameter;
- merged-weight creation, weight copying, quantization, artifact promotion, or
  Registry publication;
- serving, canary traffic, deployment, Provider, MCP, Desktop, or Runtime
  integration; or
- claims of a pristine-FP32 checkpoint, generalization, repeatability of the
  full evaluation, production safety, capacity, latency, or cost.

The base checkpoint stores BF16 values that are materialized as FP32 at load;
this is not a separately trained or pristine FP32 checkpoint. The formal
evaluation count is one, and the repository has no external execution-count
attestation. The recorded FP32 peak memory is close to its pre-registered cap
and is not a serving-capacity result.

## Failure handling

Any component-resolution failure is terminal for the attempted local package.
The validator reports the missing, unsafe, unexpected, or mismatched component
and keeps offline execution, portability, promotion, serving, and Runtime
eligibility false. It must not repair files, download an alternate revision,
change execution form, or silently continue with partial evidence.
