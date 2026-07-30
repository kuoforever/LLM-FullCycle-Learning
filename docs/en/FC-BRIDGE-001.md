# FC-BRIDGE-001: Runtime Lane A offline consumer contract

中文：[FC-BRIDGE-001.md](../FC-BRIDGE-001.md)

## Scope

`FC-BRIDGE-001` consumes the Runtime producer's Full Cycle manifest v1 and
redacted run export v1. It validates a small, explicit contract and returns a
typed object for downstream reliability/Verifier dataset work.

It is deliberately offline. It does not start the provider, MCP, desktop,
network, memory, continuation, or training layers. It does not convert Lane A
evidence into SFT text or rich multimodal episodes.

## Accepted evidence

The consumer requires:

- supported manifest and run-export versions;
- a producer-pinned Runtime commit;
- canonical manifest JSON and exact SHA-256 digest agreement;
- bounded file and event sizes;
- complete required fields with unknown fields rejected;
- `training_use=reliability_and_verifier_only`;
- redaction flags consistent with Lane A restrictions.

The fixture set includes one minimal valid export and invalid cases covering
unknown fields, unsupported versions, malformed or mismatched digests,
oversized inputs, incomplete events, and forbidden training-use values.

## Fail-closed behavior

Validation rejects before downstream use when schema, version, digest,
redaction, completeness, or size rules fail. Errors identify the violated
contract without silently coercing data or guessing missing semantics.

## Reproduction

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\validate_bridge.ps1
```

The unified repository gate is:

```powershell
python -I .\scripts\validate_offline.py
```

## Limits

Lane A excludes raw tasks, model text, tool-result bodies, screenshots,
memory, and continuation data. It can support reliability and verifier
signals, but not GUI grounding or multimodal imitation training. Rich capture
requires the separately reviewed, explicit-consent Lane B contract.

