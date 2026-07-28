"""Fail-closed, offline validation of the Runtime Lane A bridge."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, NoReturn

CONSUMER_SCHEMA_VERSION = "1.0.0"
RUNTIME_GIT_COMMIT = "8ace897f746a4aa3dd3f8b10af392ea9ba81941d"

MAX_INPUT_BYTES = 24 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 64 * 1024
MAX_EVENT_BYTES = 1024 * 1024

MANIFEST_FIELDS = frozenset(
    {
        "fullcycle_manifest_version",
        "agent_contract_version",
        "driver_contract_version",
        "trace_version",
        "checkpoint_version",
        "plan_contract_version",
        "tools",
        "automatic_export",
    }
)
SUPPORTED_VERSIONS = {
    "fullcycle_manifest_version": 1,
    "agent_contract_version": "0.1.0",
    "driver_contract_version": "1.0.0",
    "trace_version": 1,
    "checkpoint_version": 1,
    "plan_contract_version": 1,
}
AUTOMATIC_EXPORT = {
    "contains_raw_task": False,
    "contains_model_text": False,
    "contains_tool_result_text": False,
    "contains_images": False,
    "contains_memory": False,
    "contains_continuation": False,
}
TOOL_FIELDS = frozenset(
    {
        "name",
        "description",
        "input_schema",
        "effect",
        "result_content",
        "result_sensitivity",
        "redaction_policy",
        "grounding",
        "requires_host_approval",
        "invalidates_observation",
        "sensitive_arguments",
        "required_safety_baselines",
    }
)
TOOL_ENUMS = {
    "effect": frozenset({"observation", "side_effect"}),
    "result_content": frozenset({"text", "image", "text_and_image"}),
    "result_sensitivity": frozenset({"normal", "sensitive"}),
    "redaction_policy": frozenset({"none", "title_matched_only"}),
    "grounding": frozenset(
        {"none", "recent_observation", "observed_window", "ref_or_screenshot"}
    ),
}

RUN_FIELDS = frozenset(
    {
        "fullcycle_run_export_version",
        "manifest_digest",
        "run_id",
        "checkpoint",
        "events",
        "data_class",
        "training_use",
    }
)
CHECKPOINT_REQUIRED_FIELDS = frozenset(
    {
        "checkpoint_version",
        "checkpoint_sequence",
        "run_id",
        "phase",
        "policy_version",
        "recovery_status",
        "task_length",
        "observation_epoch",
        "verified_observation_epoch",
        "event_count",
        "budgets",
        "updated_at",
        "metrics",
        "resume_allowed",
        "recovery_action",
    }
)
CHECKPOINT_OPTIONAL_FIELDS = frozenset(
    {"created_at", "failure_code", "final_text_length"}
)
BUDGET_FIELDS = frozenset(
    {
        "max_model_turns",
        "max_tool_calls",
        "max_side_effects",
        "model_turns_used",
        "tool_calls_used",
        "side_effects_used",
        "max_input_tokens",
        "input_tokens_used",
    }
)
METRIC_REQUIRED_FIELDS = frozenset(
    {
        "model_calls",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "provider_latency_ms",
        "tool_latency_ms",
        "tool_failures",
        "image_results",
        "screenshot_results",
        "provider_usage_report_count",
        "retry_count",
    }
)
METRIC_OPTIONAL_FIELDS = frozenset({"run_duration_ms"})
EVENT_BASE_FIELDS = frozenset({"trace_version", "sequence", "run_id", "kind"})
EVENT_REQUIRED_FIELDS = {
    "user_task": frozenset({"task_length"}),
    "model_turn": frozenset(
        {"text_length", "tool_call_count", "input_tokens", "output_tokens", "latency_ms"}
    ),
    "tool_call": frozenset({"tool", "arguments", "redacted_fields"}),
    "tool_result": frozenset(
        {"tool", "status", "dispatch", "text_length", "image_count"}
    ),
    "observation": frozenset({"tool", "observation_epoch"}),
    "policy_decision": frozenset({"decision"}),
    "recovery": frozenset({"status"}),
}
EVENT_OPTIONAL_FIELDS = {
    "user_task": frozenset(),
    "model_turn": frozenset(),
    "tool_call": frozenset(),
    "tool_result": frozenset({"latency_ms", "code"}),
    "observation": frozenset(),
    "policy_decision": frozenset(),
    "recovery": frozenset(),
}
FORBIDDEN_RICH_FIELDS = frozenset(
    {
        "task",
        "raw_task",
        "model_text",
        "tool_result",
        "tool_result_text",
        "screenshot",
        "screenshots",
        "image",
        "images",
        "image_bytes",
        "memory",
        "continuation",
        "response",
        "content",
        "text",
    }
)
RUN_PHASES = frozenset(
    {
        "CREATED",
        "OBSERVING",
        "PLANNING",
        "WAITING_APPROVAL",
        "PAUSED",
        "EXECUTING",
        "VERIFYING",
        "SUCCESS",
        "FAILED",
        "UNKNOWN_OUTCOME",
        "CANCELLED",
    }
)
RECOVERY_STATUSES = frozenset(
    {"ready", "requires_reobservation", "unknown_outcome", "stopped"}
)
RECOVERY_ACTIONS = frozenset(
    {
        "resume_with_original_task",
        "none",
        "human_reobserve_then_start_new_run",
        "inspect_trace_then_start_new_run",
    }
)
TOOL_RESULT_STATUSES = frozenset(
    {"success", "action_error", "transport_error", "rejected", "unknown_outcome"}
)
DISPATCH_CERTAINTIES = frozenset({"not_dispatched", "dispatched", "unknown"})
POLICY_DECISIONS = frozenset(
    {"allow", "deny", "approval_required", "reobserve", "defer"}
)


class BridgeValidationError(ValueError):
    """A stable validation failure suitable for CI assertions."""

    def __init__(self, code: str, location: str, detail: str = "") -> None:
        self.code = code
        self.location = location
        self.detail = detail
        message = f"{code} at {location}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class ValidationSummary:
    consumer_schema_version: str
    runtime_git_commit: str
    manifest_digest: str
    run_id: str
    event_count: int
    data_class: str
    training_use: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Use the producer's canonical JSON algorithm."""

    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()}"


def validate_files(manifest_path: Path, run_export_path: Path) -> ValidationSummary:
    """Read two bounded regular files and validate them without opening any ports."""

    manifest = _load_bounded_json(manifest_path, "manifest")
    run_export = _load_bounded_json(run_export_path, "run_export")
    validate_manifest(manifest)
    validate_run_export(run_export, manifest)
    checkpoint = run_export["checkpoint"]
    return ValidationSummary(
        consumer_schema_version=CONSUMER_SCHEMA_VERSION,
        runtime_git_commit=RUNTIME_GIT_COMMIT,
        manifest_digest=manifest_digest(manifest),
        run_id=run_export["run_id"],
        event_count=checkpoint["event_count"],
        data_class=run_export["data_class"],
        training_use=run_export["training_use"],
    )


def validate_manifest(value: Any) -> None:
    manifest = _require_object(value, "manifest")
    _require_exact_fields(manifest, MANIFEST_FIELDS, "manifest")
    for field, expected in SUPPORTED_VERSIONS.items():
        if manifest[field] != expected or type(manifest[field]) is not type(expected):
            _fail("UNSUPPORTED_VERSION", f"manifest.{field}", repr(manifest[field]))
    if manifest["automatic_export"] != AUTOMATIC_EXPORT:
        _fail(
            "UNSAFE_AUTOMATIC_EXPORT",
            "manifest.automatic_export",
            "all six declarations must be present and false",
        )
    tools = _require_list(manifest["tools"], "manifest.tools")
    names: set[str] = set()
    for index, value in enumerate(tools):
        location = f"manifest.tools[{index}]"
        tool = _require_object(value, location)
        _require_exact_fields(tool, TOOL_FIELDS, location)
        name = _require_nonempty_string(tool["name"], f"{location}.name")
        if name in names:
            _fail("DUPLICATE_TOOL", f"{location}.name", name)
        names.add(name)
        _require_nonempty_string(tool["description"], f"{location}.description")
        _require_object(tool["input_schema"], f"{location}.input_schema")
        _validate_json_tree(tool["input_schema"], f"{location}.input_schema")
        for field, allowed in TOOL_ENUMS.items():
            if tool[field] not in allowed:
                _fail("INVALID_VALUE", f"{location}.{field}", repr(tool[field]))
        _require_bool(tool["requires_host_approval"], f"{location}.requires_host_approval")
        _require_bool(tool["invalidates_observation"], f"{location}.invalidates_observation")
        _require_string_list(tool["sensitive_arguments"], f"{location}.sensitive_arguments")
        _require_string_list(
            tool["required_safety_baselines"], f"{location}.required_safety_baselines"
        )


def validate_run_export(value: Any, manifest: Mapping[str, Any]) -> None:
    run_export = _require_object(value, "run_export")
    _require_exact_fields(run_export, RUN_FIELDS, "run_export")
    _require_exact_int(
        run_export["fullcycle_run_export_version"],
        1,
        "run_export.fullcycle_run_export_version",
        "UNSUPPORTED_VERSION",
    )
    expected_digest = manifest_digest(manifest)
    digest = run_export["manifest_digest"]
    if not isinstance(digest, str) or digest != expected_digest:
        _fail("MANIFEST_DIGEST_MISMATCH", "run_export.manifest_digest")
    run_id = _require_nonempty_string(run_export["run_id"], "run_export.run_id")
    if run_export["data_class"] != "redacted_runtime_evidence":
        _fail("INVALID_DATA_CLASS", "run_export.data_class")
    if run_export["training_use"] != "reliability_and_verifier_only":
        _fail("INVALID_TRAINING_USE", "run_export.training_use")

    checkpoint = _require_object(run_export["checkpoint"], "run_export.checkpoint")
    if len(canonical_json_bytes(checkpoint)) > MAX_CHECKPOINT_BYTES:
        _fail("CHECKPOINT_TOO_LARGE", "run_export.checkpoint")
    _validate_checkpoint(checkpoint, run_id)

    events = _require_list(run_export["events"], "run_export.events")
    if checkpoint["event_count"] != len(events):
        _fail("INCOMPLETE_EVENTS", "run_export.events", "event_count mismatch")
    if not events:
        _fail("INCOMPLETE_EVENTS", "run_export.events", "missing user_task event")
    for index, event in enumerate(events, start=1):
        if len(canonical_json_bytes(event)) > MAX_EVENT_BYTES:
            _fail("EVENT_TOO_LARGE", f"run_export.events[{index - 1}]")
        _validate_event(event, index, run_id)
    first = _require_object(events[0], "run_export.events[0]")
    if (
        first.get("kind") != "user_task"
        or first.get("task_length") != checkpoint["task_length"]
    ):
        _fail(
            "INCOMPLETE_EVENTS",
            "run_export.events[0]",
            "must be user_task matching checkpoint task_length",
        )
    _reject_rich_fields(checkpoint, "run_export.checkpoint")
    _reject_rich_fields(events, "run_export.events")


def _validate_checkpoint(checkpoint: dict[str, Any], run_id: str) -> None:
    allowed = CHECKPOINT_REQUIRED_FIELDS | CHECKPOINT_OPTIONAL_FIELDS
    _require_fields(checkpoint, CHECKPOINT_REQUIRED_FIELDS, allowed, "run_export.checkpoint")
    _require_exact_int(
        checkpoint["checkpoint_version"],
        1,
        "run_export.checkpoint.checkpoint_version",
        "UNSUPPORTED_VERSION",
    )
    if checkpoint["run_id"] != run_id:
        _fail("RUN_ID_MISMATCH", "run_export.checkpoint.run_id")
    for field in (
        "checkpoint_sequence",
        "task_length",
        "observation_epoch",
        "event_count",
    ):
        _require_nonnegative_int(checkpoint[field], f"run_export.checkpoint.{field}")
    verified = checkpoint["verified_observation_epoch"]
    if verified is not None:
        _require_nonnegative_int(
            verified, "run_export.checkpoint.verified_observation_epoch"
        )
    for field in ("policy_version", "updated_at"):
        _require_nonempty_string(checkpoint[field], f"run_export.checkpoint.{field}")
    _require_member(checkpoint["phase"], RUN_PHASES, "run_export.checkpoint.phase")
    _require_member(
        checkpoint["recovery_status"],
        RECOVERY_STATUSES,
        "run_export.checkpoint.recovery_status",
    )
    _require_member(
        checkpoint["recovery_action"],
        RECOVERY_ACTIONS,
        "run_export.checkpoint.recovery_action",
    )
    _require_bool(checkpoint["resume_allowed"], "run_export.checkpoint.resume_allowed")
    for field in ("created_at", "failure_code"):
        if field in checkpoint:
            _require_nonempty_string(checkpoint[field], f"run_export.checkpoint.{field}")
    if "final_text_length" in checkpoint:
        _require_nonnegative_int(
            checkpoint["final_text_length"], "run_export.checkpoint.final_text_length"
        )

    budgets = _require_object(checkpoint["budgets"], "run_export.checkpoint.budgets")
    _require_exact_fields(budgets, BUDGET_FIELDS, "run_export.checkpoint.budgets")
    for field, value in budgets.items():
        _require_nonnegative_int(value, f"run_export.checkpoint.budgets.{field}")
    for used, maximum in (
        ("model_turns_used", "max_model_turns"),
        ("tool_calls_used", "max_tool_calls"),
        ("side_effects_used", "max_side_effects"),
        ("input_tokens_used", "max_input_tokens"),
    ):
        if budgets[used] > budgets[maximum]:
            _fail(
                "INVALID_VALUE",
                f"run_export.checkpoint.budgets.{used}",
                f"exceeds {maximum}",
            )

    metrics = _require_object(checkpoint["metrics"], "run_export.checkpoint.metrics")
    _require_fields(
        metrics,
        METRIC_REQUIRED_FIELDS,
        METRIC_REQUIRED_FIELDS | METRIC_OPTIONAL_FIELDS,
        "run_export.checkpoint.metrics",
    )
    for field, value in metrics.items():
        _require_nonnegative_int(value, f"run_export.checkpoint.metrics.{field}")


def _validate_event(value: Any, sequence: int, run_id: str) -> None:
    location = f"run_export.events[{sequence - 1}]"
    event = _require_object(value, location)
    kind = event.get("kind")
    if not isinstance(kind, str) or kind not in EVENT_REQUIRED_FIELDS:
        _fail("UNKNOWN_EVENT_KIND", f"{location}.kind", repr(kind))
    required = EVENT_BASE_FIELDS | EVENT_REQUIRED_FIELDS[kind]
    allowed = required | EVENT_OPTIONAL_FIELDS[kind]
    _require_fields(event, required, allowed, location)
    _require_exact_int(event["trace_version"], 1, f"{location}.trace_version", "UNSUPPORTED_VERSION")
    _require_exact_int(event["sequence"], sequence, f"{location}.sequence", "INCOMPLETE_EVENTS")
    if event["run_id"] != run_id:
        _fail("RUN_ID_MISMATCH", f"{location}.run_id")

    if kind == "user_task":
        _require_nonnegative_int(event["task_length"], f"{location}.task_length")
    elif kind == "model_turn":
        for field in EVENT_REQUIRED_FIELDS[kind]:
            _require_nonnegative_int(event[field], f"{location}.{field}")
    elif kind == "tool_call":
        _require_nonempty_string(event["tool"], f"{location}.tool")
        _require_object(event["arguments"], f"{location}.arguments")
        _validate_json_tree(event["arguments"], f"{location}.arguments")
        _require_string_list(event["redacted_fields"], f"{location}.redacted_fields")
    elif kind == "tool_result":
        _require_nonempty_string(event["tool"], f"{location}.tool")
        _require_member(event["status"], TOOL_RESULT_STATUSES, f"{location}.status")
        _require_member(
            event["dispatch"], DISPATCH_CERTAINTIES, f"{location}.dispatch"
        )
        for field in ("text_length", "image_count"):
            _require_nonnegative_int(event[field], f"{location}.{field}")
        if "latency_ms" in event:
            _require_nonnegative_int(event["latency_ms"], f"{location}.latency_ms")
        if "code" in event:
            _require_nonempty_string(event["code"], f"{location}.code")
    elif kind == "observation":
        _require_nonempty_string(event["tool"], f"{location}.tool")
        _require_nonnegative_int(event["observation_epoch"], f"{location}.observation_epoch")
    elif kind == "policy_decision":
        _require_member(
            event["decision"], POLICY_DECISIONS, f"{location}.decision"
        )
    elif kind == "recovery":
        _require_member(event["status"], RECOVERY_STATUSES, f"{location}.status")


def _load_bounded_json(path: Path, label: str) -> Any:
    if not isinstance(path, Path):
        path = Path(path)
    try:
        if path.is_symlink() or not path.is_file():
            _fail("UNSAFE_INPUT_PATH", label)
        size = path.stat().st_size
        if size <= 0:
            _fail("EMPTY_JSON", label)
        if size > MAX_INPUT_BYTES:
            _fail("INPUT_TOO_LARGE", label, f"{size} bytes")
        with path.open("rb") as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
    except BridgeValidationError:
        raise
    except OSError as exc:
        raise BridgeValidationError("INPUT_READ_FAILED", label, str(exc)) from exc
    if len(raw) > MAX_INPUT_BYTES:
        _fail("INPUT_TOO_LARGE", label)
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=lambda pairs: _pairs_without_duplicates(pairs, label),
            parse_constant=lambda value: _reject_constant(value, label),
        )
    except BridgeValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BridgeValidationError("MALFORMED_JSON", label, str(exc)) from exc


def _pairs_without_duplicates(pairs: list[tuple[str, Any]], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("DUPLICATE_JSON_KEY", label, key)
        result[key] = value
    return result


def _reject_constant(value: str, label: str) -> None:
    _fail("NONFINITE_NUMBER", label, value)


def _validate_json_tree(value: Any, location: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("NONFINITE_NUMBER", location)
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_json_tree(nested, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                _fail("INVALID_TYPE", location, "object key must be string")
            _validate_json_tree(nested, f"{location}.{key}")
        return
    _fail("INVALID_TYPE", location, type(value).__name__)


def _reject_rich_fields(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{location}.{key}"
            if key.lower() in FORBIDDEN_RICH_FIELDS:
                _fail("FORBIDDEN_RICH_FIELD", child)
            _reject_rich_fields(nested, child)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_rich_fields(nested, f"{location}[{index}]")


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("INVALID_TYPE", location, "expected object")
    return value


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        _fail("INVALID_TYPE", location, "expected array")
    return value


def _require_exact_fields(value: Mapping[str, Any], expected: frozenset[str], location: str) -> None:
    _require_fields(value, expected, expected, location)


def _require_fields(
    value: Mapping[str, Any],
    required: frozenset[str],
    allowed: frozenset[str],
    location: str,
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - allowed)
    if missing:
        _fail("MISSING_FIELD", location, ",".join(missing))
    if extra:
        _fail("UNKNOWN_FIELD", location, ",".join(extra))


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("INVALID_TYPE", location, "expected non-empty string")
    return value


def _require_string_list(value: Any, location: str) -> None:
    items = _require_list(value, location)
    if any(not isinstance(item, str) or not item for item in items):
        _fail("INVALID_TYPE", location, "expected non-empty strings")
    if len(set(items)) != len(items):
        _fail("DUPLICATE_VALUE", location)


def _require_bool(value: Any, location: str) -> None:
    if not isinstance(value, bool):
        _fail("INVALID_TYPE", location, "expected boolean")


def _require_member(value: Any, allowed: frozenset[str], location: str) -> None:
    if not isinstance(value, str) or value not in allowed:
        _fail("INVALID_VALUE", location, repr(value))


def _require_nonnegative_int(value: Any, location: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail("INVALID_TYPE", location, "expected non-negative integer")


def _require_exact_int(value: Any, expected: int, location: str, code: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        _fail(code, location, repr(value))


def _fail(code: str, location: str, detail: str = "") -> NoReturn:
    raise BridgeValidationError(code, location, detail)
