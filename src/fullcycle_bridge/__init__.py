"""Strict offline consumer for Desktop Runtime Lane A evidence."""

from .consumer import (
    BridgeValidationError,
    ValidationSummary,
    canonical_json_bytes,
    load_validated_files,
    manifest_digest,
    validate_files,
    validate_manifest,
    validate_run_export,
)

__all__ = [
    "BridgeValidationError",
    "ValidationSummary",
    "canonical_json_bytes",
    "load_validated_files",
    "manifest_digest",
    "validate_files",
    "validate_manifest",
    "validate_run_export",
]
