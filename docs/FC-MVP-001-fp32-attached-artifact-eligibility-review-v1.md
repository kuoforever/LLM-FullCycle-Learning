# FC-MVP-001 FP32 Attached Artifact Eligibility Review v1 Evidence

Status: completed locally on 2026-08-05; frozen fixed-compiler quality evidence
is favorable, but the current offline artifact package is incomplete and
ineligible. Promotion and Runtime integration remain prohibited.

## Result

The review classifies the current state as
`fp32_attached_fixed_compiler_favorable_eval_but_offline_artifact_package_incomplete`.
The frozen repository-local evidence remains usable, but the deliverable is not
an independently specified offline artifact:

| Decision | Result |
|---|---:|
| Fixed-compiler compiled quality evidence favorable | `true` |
| Repository-local evidence usable | `true` |
| Offline artifact eligible | `false` |
| Portable package eligible | `false` |
| Preferred offline candidate | `false` |
| Serving readiness established | `false` |
| Artifact promotion allowed | `false` |
| Merged artifact allowed | `false` |
| Runtime eligible | `false` |

This is a validated negative decision, not a failed review. It separates the
quality of the frozen fixed-compiler result from the completeness and authority
of the package that would be needed to reproduce it elsewhere.

## Benefit, dependency, and cost boundary

The review consumes the already frozen remediation evaluation without running
the model or evaluation again. Its comparison basis is the fixed-compiler
compiled outputs:

- argument exact match improves from `0.20` to `0.25`;
- argument field F1 improves from `0.2608695652173913` to
  `0.29787234042553196`;
- the only strict per-example improvement is `eval-016.arguments`;
- tool accuracy and risk macro F1 remain `0.95` and
  `0.7095238095238096`;
- compiled safety checks pass and there are zero compiled per-example
  regression events; and
- only one registered 20-case full-eval run exists, so full-eval repeatability
  and variance are not estimated.

The benefit cannot be attributed to the bare FP32 Adapter. Raw semantic
validity falls from the BF16 reference `0.85` to FP32 `0.80`; the fixed
`compile_decision` v1 changes `eval-001`, `eval-009`, `eval-014`, and
`eval-020` and restores compiled semantic validity to `1.0`. The compiler is
therefore a required part of the reviewed candidate identity, but it is not
bound by the current Adapter package.

The FP32 run peaks at `6,267,895,296` allocated GPU bytes versus
`3,150,315,520` for BF16, a ratio of `1.9896087411587269`. It remains below the
pre-registered cap of `6,300,631,040` by only `32,735,744` bytes. Passing that
prospective cap does not establish serving capacity or a stable speedup.

## Adapter integrity audit

The frozen Adapter directory is exact and remains unchanged:

| File | Bytes | SHA-256 |
|---|---:|---|
| `adapter_config.json` | `793` | `8eb104c3af2f4deb3abe5e471b3d3a74cb306683c1fdadb95488de981ba14c16` |
| `adapter_model.safetensors` | `17,462,432` | `efb62471e105b8ef25641200967d447b8cc2f3ff565937bc47193fbf79f4f342` |
| `README.md` | `5,107` | `353053cad9659d849cbf1fdacc7d9b86b82fb72197e2d101785843a4109bc522` |

The safetensors header and payload audit finds:

- `224` F32 tensors and `4,358,144` parameters;
- `28` layers and `112` LoRA target modules;
- `112` A matrices of shape `16 x 1536`;
- `56` Q/O B matrices of shape `1536 x 16` and `56` K/V B matrices of shape
  `256 x 16`; and
- contiguous data offsets covering exactly `17,432,576` payload bytes.

Two prior fresh FP32 attached canary loads reproduce the same token and output
digests. This supports repository-local behavior evidence for the one frozen
canary; it is not a clean-checkout package-reproducibility result or a repeated
full-eval result.

## Six packaging blockers

Eligibility uses one categorical rubric for favorable and neutral upstream
outcomes. All six requirements must pass; benefit cannot waive an integrity or
packaging requirement.

| Blocking finding | Frozen observation |
|---|---|
| `base_model_revision_binding_missing` | Adapter metadata has `revision=null`; no authoritative composite sidecar exists. |
| `composite_manifest_missing` | The immutable three-file Adapter directory contains no composite package manifest, and this review accepts no sidecar input. |
| `package_use_and_limitations_documentation_incomplete` | The Adapter README retains `39` template placeholders and does not document the reviewed use and limits. |
| `portable_base_model_binding_missing` | `base_model_name_or_path` is the machine-relative `work\models\Qwen2.5-1.5B-Instruct`, which has no package authority. |
| `required_compiler_binding_missing` | The Adapter package does not bind the required compiler file and symbol hashes. |
| `tokenizer_file_manifest_missing` | Repo/revision are pinned upstream, but no authoritative file-level tokenizer manifest belongs to the package. |

The reviewer never treats an `artifact_manifest.json` filename, README
substring, or edited Adapter metadata as proof. A future positive decision must
come from the next gate's independently parsed external metadata sidecar while
the three Adapter files remain immutable.

## Frozen review artifact and provenance

The contract and builder were frozen before artifact generation in commit
`a36cc965531cef781cd66aff3c0ff4c481d56520`. The generated review is
[`baseline/fc-mvp-001-fp32-attached-artifact-eligibility-review-v1.json`](../baseline/fc-mvp-001-fp32-attached-artifact-eligibility-review-v1.json):

| Property | Value |
|---|---|
| Bytes | `15,278` |
| File SHA-256 | `81977f318c6bcfed8d3844575dc245d4b94c2636a2359165f9aa5553c9b006f8` |
| Internal report digest | `285d5e5e25dfd16de5adc6cb760fe54588af68d8580308b54ccfaf612d51636b` |
| Direct source roots | `25` |

Each source is read into one immutable byte payload. The same bytes are used to
derive its SHA-256 and, where applicable, the parsed JSON/text, Adapter
manifest, safetensors audit, and canonical eval digest. The pure validator then
requires payload, parsed content, observed hash, and the external roots
committed by the hard-coded review-file SHA to agree. This closes both
object/hash substitution and multi-read time-of-check/time-of-use gaps.

The 25 direct roots cover the review inputs, builder, contract, and their direct
`consumer.py`, `tool_router_sft.py`, and `tool_router.py` dependencies. Earlier
evaluation and training closure remains transitively bound by the frozen
pre-registration, remediation gate, SFT config, training evidence, and
lifecycle evidence; the review does not falsely claim to duplicate every
upstream raw file as a direct root.

Mutation tests reject external-root drift, parsed objects that differ from raw
payloads, ignored extra fields, generation/gate changes, missing canary digests,
tensor-audit changes, non-finite numbers, and a forged eligibility decision
even when its internal report digest is recomputed.

## Validation

The unified offline gate reports `valid=true`, `268` tests, and `30` audited
source files on CPython 3.11.15, 3.12.12, and 3.13.7. Ruff, scoped mypy 2.3.0,
py_compile, builder `--check`, and `git diff --check` also pass. These checks
load no model, use no GPU or network, and do not consume another formal-eval
run.

```powershell
python .\scripts\review_tool_router_fp32_attached_artifact_eligibility.py --check
python -I .\scripts\validate_offline.py
```

## Single next objective

The next gate is
`FC-MVP-001-fp32-attached-offline-package-manifest-v1`. It may create and
validate one metadata-only composite manifest that binds the unchanged Adapter,
pinned base and tokenizer file identities, required compiler, prompt,
generation and precision policies, environment, and attached-only execution.

It must not rerun the full evaluation; add data; train; tune against eval
answers; change the compiler, prompt, generation, or execution form; mutate or
copy model/Adapter/tokenizer files; merge or save weights; promote an artifact;
deploy serving; or integrate Runtime/Provider/MCP/Desktop. Completing that
manifest gate would not by itself establish clean-checkout behavioral
reproducibility, full-eval repeatability, serving readiness, or Runtime
eligibility.
