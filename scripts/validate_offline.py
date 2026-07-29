"""Self-contained FC-MVP-000 validation gate for Python 3.11-3.13."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import tomllib
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
BASELINE_PATH = ROOT / "baseline" / "fc-mvp-000.json"
TOOL_ROUTER_BASELINE_PATH = ROOT / "baseline" / "fc-mvp-001-schema-eval.json"
SUPPORTED_MINORS = {(3, 11), (3, 12), (3, 13)}
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "computer_use_agent",
        "computer_use_mcp",
        "httpx",
        "mcp",
        "requests",
        "socket",
        "urllib",
        "webbrowser",
    }
)


class GateError(RuntimeError):
    pass


def main() -> int:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPATH"] = str(SRC)
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(ROOT))

    version = (sys.version_info.major, sys.version_info.minor)
    if version not in SUPPORTED_MINORS:
        raise GateError(f"unsupported Python minor: {version[0]}.{version[1]}")

    baseline = _load_json(BASELINE_PATH)
    _validate_project_metadata(baseline)
    _validate_artifact_hashes(baseline)
    audited_files = _audit_import_boundary()
    bridge_summary, record_count, router_summary = _validate_fixed_outputs()
    tests_run = _run_tests()

    result = {
        "valid": True,
        "baseline_version": baseline["baseline_version"],
        "source_code_commit": baseline["source_code_commit"],
        "python": sys.version.split()[0],
        "implementation": sys.implementation.name,
        "runtime_dependencies": 0,
        "artifact_hashes_verified": len(baseline["artifacts"]),
        "source_files_audited": audited_files,
        "tests_run": tests_run,
        "manifest_digest": bridge_summary.manifest_digest,
        "dataset_records": record_count,
        "tool_router_seed_records": router_summary["seed_records"],
        "tool_router_eval_records": router_summary["eval_records"],
        "tool_router_eval_digest": router_summary["eval_digest"],
        "tool_router_dangerous_false_approvals": router_summary["baseline"][
            "dangerous_false_approvals"
        ],
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _validate_project_metadata(baseline: dict[str, Any]) -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    expected_minors = [f"{major}.{minor}" for major, minor in sorted(SUPPORTED_MINORS)]
    if project["requires-python"] != baseline["python_requires"]:
        raise GateError("python_requires does not match baseline")
    if project["version"] != baseline["package_version"]:
        raise GateError("package version does not match baseline")
    if project["dependencies"] != baseline["runtime_dependencies"]:
        raise GateError("runtime dependencies do not match baseline")
    if baseline["required_python_minors"] != expected_minors:
        raise GateError("Python matrix does not match gate")
    lock_lines = [
        line.strip()
        for line in (ROOT / "requirements" / "runtime.lock")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if lock_lines:
        raise GateError("runtime.lock must remain empty for the stdlib baseline")


def _validate_artifact_hashes(baseline: dict[str, Any]) -> None:
    artifacts = baseline.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise GateError("baseline artifacts are missing")
    for relative, expected in artifacts.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise GateError(f"unsafe or missing baseline artifact: {relative}")
        actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise GateError(f"baseline artifact digest mismatch: {relative}")


def _audit_import_boundary() -> int:
    count = 0
    for path in sorted((SRC / "fullcycle_bridge").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count += 1
        for node in ast.walk(tree):
            imported: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.partition(".")[0]
                    if root in FORBIDDEN_IMPORT_ROOTS:
                        raise GateError(f"forbidden import {root} in {path.name}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported = node.module.partition(".")[0]
            if imported in FORBIDDEN_IMPORT_ROOTS:
                raise GateError(f"forbidden import {imported} in {path.name}")
    return count


def _validate_fixed_outputs() -> tuple[Any, int, dict[str, Any]]:
    from fullcycle_bridge import validate_files
    from fullcycle_bridge.consumer import canonical_json_bytes
    from fullcycle_bridge.dataset import map_many
    from fullcycle_bridge.tool_router import (
        baseline_predict,
        evaluate,
        fixture_digest,
        load_fixture,
    )

    manifest = ROOT / "fixtures" / "bridge_v1" / "valid" / "runtime-manifest.json"
    minimal = ROOT / "fixtures" / "bridge_v1" / "valid" / "minimal-run-export.json"
    input_root = ROOT / "fixtures" / "reliability_dataset_v1" / "inputs"
    inputs = [
        input_root / "failure-denial-recovery-budget-sequence.json",
        input_root / "unknown-outcome.json",
    ]
    summary = validate_files(manifest, minimal)
    records = map_many(manifest, inputs)
    actual = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    expected = (
        ROOT / "fixtures" / "reliability_dataset_v1" / "expected-records.jsonl"
    ).read_bytes()
    if actual != expected:
        raise GateError("dataset JSONL differs from the frozen fixture")
    router_baseline = _load_json(TOOL_ROUTER_BASELINE_PATH)
    _validate_named_hashes(router_baseline["artifact_hashes"])
    seed = load_fixture(ROOT / "fixtures" / "tool_router_v1" / "seed.json")
    evaluation = load_fixture(ROOT / "fixtures" / "tool_router_v1" / "eval.json")
    router_summary = {
        "seed_records": len(seed),
        "eval_records": len(evaluation),
        "seed_digest": fixture_digest(seed),
        "eval_digest": fixture_digest(evaluation),
        "baseline": evaluate(
            evaluation, [baseline_predict(record) for record in evaluation]
        ),
    }
    for key in ("seed_records", "eval_records", "seed_digest", "eval_digest"):
        if router_summary[key] != router_baseline[key]:
            raise GateError(f"Tool Router baseline mismatch: {key}")
    if router_summary["baseline"] != router_baseline["deterministic_rule_baseline"]:
        raise GateError("Tool Router metrics differ from the frozen baseline")
    return summary, len(records), router_summary


def _validate_named_hashes(artifacts: object) -> None:
    if not isinstance(artifacts, dict) or not artifacts:
        raise GateError("named artifact hashes are missing")
    for relative, expected in artifacts.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise GateError(f"unsafe or missing artifact: {relative}")
        actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise GateError(f"artifact digest mismatch: {relative}")


def _run_tests() -> int:
    suite = unittest.defaultTestLoader.discover(str(TESTS))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise GateError("unit test suite failed")
    return result.testsRun


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"expected object: {path}")
    return value


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(
            json.dumps(
                {"valid": False, "code": "FC_MVP_000_GATE_FAILED", "detail": str(exc)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
