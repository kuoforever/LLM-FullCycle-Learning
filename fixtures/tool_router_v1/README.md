# Tool Router v1 frozen fixtures

`seed.json` and `eval.json` are reviewed, fully offline gold fixtures for the
`FC-MVP-001` schema/eval gate. Each file contains two examples for each of the
ten required scenario categories. The eval answers are content-pinned by
canonical SHA-256 in `baseline/fc-mvp-001-schema-eval.json`.

The files intentionally contain no Provider output, MCP payload, Desktop
observation, network result, Memory, Continuation, screenshot, raw trace, or
tool-result content. They describe candidate routing decisions only; they do
not authorize or execute a tool.

`scripts/build_tool_router_fixtures.py` is the deterministic, reviewed source
for these 40 concrete records. Changing the builder does not silently change
the gate: regenerate the files, review the diff, and explicitly update the
frozen hashes and metrics.

This is a schema/evaluation seed, not the 200-500 record training dataset.
Expansion must preserve the frozen eval set and split new task families before
any model result is inspected.
