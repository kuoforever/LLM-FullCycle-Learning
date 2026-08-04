"""Pure classification for the Tool Router FP32 merge remediation gate."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, NoReturn

from .tool_router import ToolRouterValidationError

MERGE_REMEDIATION_VERSION = 1


def analyze_candidate_runs(
    candidate_runs: Iterable[Iterable[int]],
    *,
    reference_token_count: int,
    reference_token_ids_sha256: str,
) -> dict[str, Any]:
    """Classify two fresh FP32 candidate runs against a frozen token digest."""

    candidates = [
        _token_run(run, f"$.candidate_runs[{index}]")
        for index, run in enumerate(candidate_runs)
    ]
    if len(candidates) != 2:
        _fail("INVALID_CANDIDATE_REPEAT_COUNT", "$.candidate_runs", str(len(candidates)))
    if (
        not isinstance(reference_token_count, int)
        or isinstance(reference_token_count, bool)
        or reference_token_count <= 0
    ):
        _fail(
            "INVALID_REFERENCE_TOKEN_COUNT",
            "$.reference_token_count",
            repr(reference_token_count),
        )
    if not _is_canonical_sha256(reference_token_ids_sha256):
        _fail(
            "INVALID_REFERENCE_TOKEN_DIGEST",
            "$.reference_token_ids_sha256",
            repr(reference_token_ids_sha256),
        )

    token_counts = [len(run) for run in candidates]
    token_digests = [token_ids_sha256(run) for run in candidates]
    repeats_identical = candidates[0] == candidates[1]
    count_matches = all(count == reference_token_count for count in token_counts)
    digest_matches = all(
        digest == reference_token_ids_sha256 for digest in token_digests
    )
    reference_identity = repeats_identical and count_matches and digest_matches
    if not repeats_identical:
        classification = "fp32_candidate_within_path_nondeterminism"
    elif reference_identity:
        classification = "fp32_safe_merge_output_identity_restored"
    else:
        classification = "deterministic_fp32_merge_output_drift"
    return {
        "candidate_repeats_identical": repeats_identical,
        "candidate_token_counts": token_counts,
        "candidate_token_ids_sha256": token_digests,
        "reference_token_count_match": count_matches,
        "reference_token_digest_match": digest_matches,
        "independent_bf16_reference_identity": reference_identity,
        "classification": classification,
    }


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    """Hash a non-empty token sequence using the gate's canonical encoding."""

    run = _token_run(token_ids, "$.token_ids")
    payload = ",".join(str(value) for value in run).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _token_run(value: Iterable[int], path: str) -> list[int]:
    run = list(value)
    if not run or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0
        for token in run
    ):
        _fail("INVALID_TOKEN_RUN", path, repr(run))
    return run


def _is_canonical_sha256(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = [
    "MERGE_REMEDIATION_VERSION",
    "analyze_candidate_runs",
    "token_ids_sha256",
]
