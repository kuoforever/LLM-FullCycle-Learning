# Environment baseline

> Baseline date: 2026-07-28  
> Scope: `FC-MVP-000` bridge and Reliability/Verifier Dataset v1 only

## Frozen runtime environment

The bridge baseline is deliberately standard-library-only:

```text
Python: >=3.11,<3.14
Preferred local minor: 3.12.12
Runtime dependencies: none
Package version: 0.2.0
Consumer schema: 1.0.0
Reliability dataset schema: 1
```

`requirements/runtime.lock` contains no installable lines. `pyproject.toml`
must retain `dependencies=[]`; the offline gate rejects drift in either file.
Training, Serving and Runtime environments remain separate future environments
and are not represented by this lock.

## Local Python matrix evidence

The same command was executed with isolated mode:

```powershell
<python> -I .\scripts\validate_offline.py
```

| Python | Interpreter source | Tests | Artifact hashes | Dataset records | Result |
|---|---|---:|---:|---:|---|
| 3.11.15 | Task-local Conda prefix under ignored `work/python-matrix/conda311` | 21 | 7 | 2 | Pass |
| 3.12.12 | Existing isolated Conda environment `torch-gpu` | 21 | 7 | 2 | Pass |
| 3.13.7 | Windows system CPython | 21 | 7 | 2 | Pass |

The 3.11 interpreter was provisioned online once because no local 3.11 runtime
or cached package existed. The validation command itself installs nothing and
uses no network, provider, MCP, Desktop, Approval, Memory or Continuation
integration.

## Static validation tools

Static checks are development-only and are not runtime dependencies:

```text
ruff 0.15.22
mypy 2.3.0
```

They are currently supplied by the sibling Runtime repository's existing
development environment. The standard-library gate remains independently
runnable without them.

## CI status

`.github/workflows/offline-baseline.yml` defines the identical gate for Python
3.11, 3.12 and 3.13 without installing project dependencies. This repository
has no configured remote, so no GitHub Actions run exists yet. The workflow is
configured but must not be described as CI-passed until a remote is explicitly
chosen and the run completes.

## Training environment boundary

PyTorch/CUDA, QLoRA, Transformers, vLLM and GPU stability are intentionally not
part of this baseline. They belong to later, separately locked training and
serving environments; the present evidence must not be used to claim those
environments are ready.
