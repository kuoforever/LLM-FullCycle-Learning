# FC-MVP-001 FP32 attached offline package manifest v1

## Outcome

`FC-MVP-001-fp32-attached-offline-package-manifest-v1` completed locally on
2026-08-06. The gate creates and strictly validates one external metadata-only
composite manifest for the unchanged FP32 attached LoRA package. Strict
recomputation derives the classification
`fp32_attached_metadata_only_composite_manifest_complete`.

The contract and builder were frozen before artifact generation at commit
`60d28be26436bc616e874692c4624d9d38a0d7a5`. The generated manifest is
`baseline/fc-mvp-001-fp32-attached-offline-package-manifest-v1.json`, is
17,487 bytes, and has raw-file SHA-256
`4125f2eef2a4b8f07015169ac7fb77b830514e053a4624aa703e5f5a64943eb0`.
The builder's exclusive-create operation and its subsequent `--check`
reconstruction produced the same bytes and hash.

## Bound package identity

The manifest binds 12 component files totaling 3,116,440,260 bytes:

- five base-model files, including the 3,087,467,144-byte
  `model.safetensors` file;
- four tokenizer files;
- the exact unchanged three-file Adapter, including 224 F32 tensors and
  4,358,144 parameters.

The base and tokenizer identity is `Qwen/Qwen2.5-1.5B-Instruct` at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The manifest also binds the
required decision compiler file, `compile_decision` symbol, direct compiler
dependencies, Adapter inspector and its direct dependencies, exact prompt,
effective generation and precision contract, environment lock and recorded
environment, and the structured use/limitations contract.

Eighteen direct source roots are hashed. Fifteen of those are fixed
repository-relative paths; the remaining three are the Adapter files. The
Adapter's recorded historical machine-relative base path remains present as a
non-authoritative fact. No absolute or caller-supplied machine path is embedded
as resolution authority.

## External trust and fail-closed resolution

The manifest is factual data. It deliberately contains no self-digest,
`passed` or `eligible` decision, Runtime decision, or next-gate decision.
`scripts/validate_offline.py` supplies two independent trust roots instead:

1. the exact raw-file manifest SHA-256; and
2. an exact hard-coded map of all 18 expected source hashes.

The validator authenticates one stable raw-byte manifest snapshot before
strictly rebuilding the expected object from independently read source
payloads. A caller cannot replace the repository source map or derive expected
hashes from the manifest. Duplicate JSON keys, non-finite values, changed
source objects, changed source bytes, or a recomputed but altered manifest fail
closed.

Component resolution accepts only caller-supplied roots and exact regular-file
sets. Symlinks, Windows reparse points, case collisions, unexpected files or
directories, missing files, byte-count drift, hash drift, and path/handle
identity changes fail closed. Builder and resolver receipts are checked again
after all component reads so an earlier file cannot be silently replaced while
a later large file is being hashed.

On the current machine, exact-root resolution succeeds for base plus tokenizer
`9/9`, Adapter `3/3`, and repository sources `15/15`. This proves that the
bytes at those supplied roots match the manifest. It is not a clean-location
attestation. In a clean GitHub checkout where `work/models` is absent, metadata
validation still succeeds while local component resolution and reproducibility
test eligibility correctly remain false; tracked Adapter and repository roots
must still resolve exactly.

## Decision boundary

The gate derives:

- `metadata_complete=true`;
- `offline_package_identity_complete=true`;
- `attached_package_identity_bound=true`;
- all six prior package blockers resolved; and
- eligibility to enter a later clean-location reproducibility test when the
  exact caller-supplied roots resolve.

Three blocking findings remain exactly:

- `behavioral_reproducibility_unverified`;
- `clean_location_resolution_unverified`;
- `remote_revision_origin_unverified`.

Therefore behavioral reproducibility and remote revision-origin attestation
remain false. Offline-artifact eligibility, portable-package eligibility,
preferred-candidate status, serving readiness, artifact promotion, merged
artifact permission, and Runtime eligibility all remain false.

No full evaluation or model generation ran in this gate. No data, training,
tuning, compiler, prompt, generation, precision, model, Adapter, tokenizer,
weight, serving, Runtime, Provider, MCP, or Desktop integration changed. No
tracked model, Adapter, or tokenizer file was mutated or copied into the
package.

## Validation evidence

The final unified offline gate passes with `valid=true`, `295 tests`, and
`source_files_audited=31` on CPython 3.11.9, 3.12.12, and 3.13.7. The output
reports the manifest SHA, metadata and package identity completeness, current
local resolution, all three remaining blockers, and the exact next gate.

The following checks also pass:

```powershell
python scripts\build_tool_router_fp32_attached_offline_package_manifest.py --check
python -I scripts\validate_offline.py
ruff check scripts\validate_offline.py scripts\build_tool_router_fp32_attached_offline_package_manifest.py src\fullcycle_bridge\tool_router_fp32_attached_offline_package_manifest.py tests\test_tool_router_fp32_attached_offline_package_manifest.py
mypy --strict src\fullcycle_bridge\tool_router_fp32_attached_offline_package_manifest.py scripts\build_tool_router_fp32_attached_offline_package_manifest.py
python -m py_compile scripts\validate_offline.py scripts\build_tool_router_fp32_attached_offline_package_manifest.py src\fullcycle_bridge\tool_router_fp32_attached_offline_package_manifest.py tests\test_tool_router_fp32_attached_offline_package_manifest.py
git diff --check
```

An independent read-only review found no remaining freeze blocker after
checking the 27 focused tests, the real 3,087,467,144-byte streaming build,
the three exact resolution groups, external trust roots, and whole-build file
receipts.

## Next gate

The single next objective is
`FC-MVP-001-fp32-attached-offline-package-reproducibility-v1`. It must freeze a
clean-location materialization, resolution, execution, and comparison protocol
before model execution. Manifest validity and current-machine exact-root
resolution are entry evidence only; neither is behavioral reproducibility,
artifact promotion, or Runtime evidence.
