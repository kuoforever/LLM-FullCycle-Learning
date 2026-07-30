# Tool Router v1 frozen fixtures

`seed.json` and `eval.json` are reviewed, fully offline gold fixtures for the
`FC-MVP-001` schema/eval gate. Each file contains two examples for each of the
ten required scenario categories. The eval answers are content-pinned by
canonical SHA-256 in `baseline/fc-mvp-001-schema-eval.json`.

`train.json` contains 160 records and `validation.json` contains 40 records.
Each category contributes 16 train and four validation examples.
`family-manifest.json` assigns the 200 records to 60 reviewed task families:
40 train families with four variants each and 20 validation families with two
variants each. Family IDs never cross splits.

The files intentionally contain no Provider output, MCP payload, Desktop
observation, network result, Memory, Continuation, screenshot, raw trace, or
tool-result content. They describe candidate routing decisions only; they do
not authorize or execute a tool.

`scripts/build_tool_router_fixtures.py` is the deterministic, reviewed source
for the original 40 seed/eval records.
`scripts/build_tool_router_dataset.py` is the reviewed source for train,
validation, and family-manifest artifacts. Changing either builder does not
silently change the gate: regenerate the files, review the diff, and explicitly
update the frozen hashes and metrics.

The data-expansion gate audits exact duplicates, cross-split family overlap,
cross-split token Jaccard similarity above `0.8`, category/risk/tool/result
distribution, schema validity, and dangerous false approvals. The frozen eval
file remains byte-for-byte unchanged.
