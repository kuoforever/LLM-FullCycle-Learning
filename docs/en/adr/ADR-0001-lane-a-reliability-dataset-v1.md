# ADR-0001: Lane A Reliability/Verifier Dataset v1

中文：[ADR-0001](../../adr/ADR-0001-lane-a-reliability-dataset-v1.md)

## Context

Runtime Lane A exports redacted operational evidence. The dataset mapper needs
useful reliability and verifier supervision without inventing semantic labels
that the export cannot support.

## Decision

Create a strict versioned dataset schema and deterministic mapper from an
accepted bridge bundle to canonical JSONL. Version 1 emits only signals
derivable from Runtime facts:

- failure and unknown-outcome state;
- policy denial;
- recovery evidence;
- budget-limit evidence;
- ordered tool sequence and outcome features.

The mapper does not generate natural-language SFT examples, retry or rollback
labels requiring interpretation, raw user content, tool-result bodies,
screenshots, memory, continuation, or rich multimodal episodes.

## Deterministic labeling

Labels are computed from explicit event/state fields and fixed precedence
rules. Unknown or unsupported evidence remains unknown rather than being
inferred. Canonical serialization makes exact byte comparison and artifact
hashing possible.

## Rejected alternatives

- Training directly from raw traces: violates the redacted Lane A boundary.
- Inferring intent or success from sparse events: produces unsupported labels.
- Using model self-judgment as ground truth: is not an independent verifier.
- Mixing rich capture into the safety export: weakens consent and retention
  boundaries.

## Consequences

The dataset is narrow but auditable, deterministic, and safe for reliability
and verifier work. It cannot train GUI grounding or behavioral imitation.
Those tasks require Lane B with explicit consent, visible capture, redaction,
retention/deletion controls, and state-based success verification.

