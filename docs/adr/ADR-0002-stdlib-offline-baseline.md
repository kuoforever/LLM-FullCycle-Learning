# ADR-0002：Standard-library offline baseline

- Status: Accepted
- Date: 2026-07-28
- Task: `FC-MVP-000`

## Context

The first executable Full Cycle baseline only validates and maps bounded JSON.
Adding a dataframe, schema runtime or packaging dependency would make clean
offline verification harder without improving the current safety contract.
The repository also has no remote, so local and future CI must use exactly the
same entry point.

## Decision

1. Support CPython 3.11, 3.12 and 3.13 exactly.
2. Keep runtime dependencies empty and record that fact in both
   `pyproject.toml` and `requirements/runtime.lock`.
3. Use `python -I scripts/validate_offline.py` as the authoritative baseline
   gate.
4. Content-pin seven contract/schema/fixture artifacts in
   `baseline/fc-mvp-000.json`.
5. Audit package imports with the Python AST and fail if network, Provider,
   Runtime or MCP roots are introduced.
6. Run unit tests, strict bridge validation and exact dataset JSONL comparison
   in the same process.
7. Configure a 3.11-3.13 GitHub Actions matrix, while reporting it as
   unexecuted until a remote exists.

## Alternatives rejected

- Installing the package before validation: unnecessary for a src-layout
  standard-library gate and would introduce build-tool availability.
- Depending on `jsonschema` at runtime: the mapper owns the emitted structure;
  the machine-readable schema is validated separately during development.
- Reusing training or Runtime environments: it would couple unrelated
  site-packages and make the baseline less reproducible.
- Claiming workflow configuration as CI evidence: configuration is not an
  executed result.

## Consequences

The baseline is small, offline after interpreter provisioning, and directly
portable to CI. Static tools and future model stacks still require their own
locks. A remote choice and first successful Actions run remain external work
that cannot be completed by guessing a destination or pushing without
authorization.
