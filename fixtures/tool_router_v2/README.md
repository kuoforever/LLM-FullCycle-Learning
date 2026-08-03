# Tool Router safety-repair dataset v2

This directory contains the combined v2 train and validation data for the
`FC-MVP-001` safety-repair data gate. The first 160 train and 40 validation
records are the byte-stable logical v1 records. The appended 16 train and 8
validation records are reviewed hard-negative families derived from the frozen
SFT v1 badcase taxonomy.

The increment targets four observed failure classes: a dangerous action
candidate, inconsistent rejection, capability gaps misrouted as clarification,
and conflicting decision flags. Each target has one train family and a distinct
validation family. No family crosses splits.

The frozen 20-record eval is not copied or changed. The gate rejects exact and
near instruction leakage against eval, requires the original v1 data and
families to remain the exact prefix, and requires zero dangerous action
candidates and zero dangerous false approvals before a v2 training config may
be locked.

`scripts/build_tool_router_safety_v2.py` is the reviewed deterministic source.
`baseline/fc-mvp-001-safety-repair-badcases-v1.json` owns provenance and
classification. No Provider, Runtime, MCP, Desktop, Memory, Continuation,
network, or Lane B path is opened by this data-only gate.
