# FC-MVP-001 Tool Router schema/eval gate

## Outcome

This gate defines Tool Router gold record version 1, freezes a balanced
20-record evaluation set, validates 20 reviewed seed records, and records one
deterministic rule baseline. It is deliberately pre-training and offline.

The router produces only a candidate decision. Runtime Policy, Approval, WAL,
Grounding, budgets, and the sole Desktop boundary remain authoritative.

## Contract

Each record has exactly:

- `schema_version`, `example_id`, `split`, `category`, and `instruction`;
- the bounded set of `available_tools`;
- delivery, duplicate, tool-failure, loop-limit, and approval state;
- normalized `selected_tool`, scalar `arguments`, `risk_level`,
  `requires_approval`, `should_reject`, `should_fallback`, and
  `expected_result`.

The JSON Schema closes every object. The standard-library validator additionally
enforces cross-field behavior:

- selected tools must be available;
- rejection, fallback, clarification, and approval flags must agree with the
  selected tool and expected result;
- dangerous requests and duplicate deliveries must be rejected;
- known tool failure and exhausted loop budgets must fall back;
- rich fields such as Provider output, Memory, Continuation, screenshots, raw
  traces, or tool-result content are forbidden.

The input is limited to 1 MiB, 1,000 records, 2,000 instruction characters, and
20 scalar argument fields per record. Unknown versions, fields, tools, enums,
malformed JSON, duplicate IDs, nested argument content, and semantic
contradictions fail closed.

## Frozen data and metrics

`fixtures/tool_router_v1/seed.json` and `eval.json` contain two cases for each:

```text
normal tool use, missing arguments, ambiguity, dangerous request, approval,
rejection, fallback, tool failure, duplicate delivery, loop limit
```

The canonical fixture digests, raw artifact hashes, category counts, and
baseline metrics are pinned in `baseline/fc-mvp-001-schema-eval.json`.
The baseline reads only instruction, available tools, and deterministic state;
it does not read the gold category or decision. Its empty-argument policy is
intentional, so argument exact match and field F1 remain an honest zero.

## Reproduction

Run the complete repository gate:

```powershell
python -I .\scripts\validate_offline.py
```

Run only the Tool Router report from a source checkout:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m fullcycle_bridge.tool_router_cli `
  --seed .\fixtures\tool_router_v1\seed.json `
  --eval .\fixtures\tool_router_v1\eval.json
```

No command opens Provider, MCP, Desktop, network, Approval, Memory,
Continuation, training, or Runtime execution.

## Limitations and next gate

The following data-expansion gate now adds 160 train and 40 validation records
without changing the 20-record eval fixture. Sixty explicit task families are
split 40/20 between train and validation, and the offline audit rejects family
overlap, exact instruction duplicates, cross-split instruction Jaccard above
`0.8`, manifest drift, and dangerous false approvals. The observed maximum
cross-split Jaccard is `0.4166666666666667`.

Reproduce the data audit:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m fullcycle_bridge.tool_router_dataset_cli `
  --train .\fixtures\tool_router_v1\train.json `
  --validation .\fixtures\tool_router_v1\validation.json `
  --eval .\fixtures\tool_router_v1\eval.json `
  --family-manifest .\fixtures\tool_router_v1\family-manifest.json
```

This is still not the complete MVP-1 loop: it runs no model and imports no real
rich Agent trace. The next gate must lock a separate inference environment and
establish prompt-only/base-model results against the unchanged eval set before
QLoRA.
