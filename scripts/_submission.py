"""Keep Slurm submission identities outside disposable calculation workspaces."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from _configuration import hdf5_entry_path


def submission_path(output_uri: str) -> Path:
    """Locate the durable submission record beside the intended output entry."""
    output = hdf5_entry_path(output_uri)
    return output.with_name(f"{output.name}.submission.json")


def claim_submission(path: Path, record: dict[str, Any]) -> None:
    """Publish a complete initial record atomically, refusing an existing run."""
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"Submission record already exists: {path}. "
                "Inspect that run and choose a new output.uri for a new submission."
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
