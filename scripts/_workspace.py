"""Manage intermediate files on the same filesystem as the final IMAS entry."""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

from _configuration import Config, hdf5_entry_path, save_config

_TEMPORARY_MARKER = ".iter-radiation-temporary"
_TEMPORARY_PREFIX = ".iter-radiation-run-"


def write_json(path: Path, value: object) -> None:
    """Replace a JSON record atomically so jobs never read a partial record."""
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def create_run(
    config: Config,
    geometry: dict,
    components: list[str],
    *,
    directory: Path | None = None,
    validation: bool = False,
) -> tuple[Config, Path]:
    """Freeze all intended components before any worker starts."""
    if not components or len(set(components)) != len(components):
        raise ValueError("A run requires unique nonempty component names")
    if hdf5_entry_path(config.output.uri).exists():
        raise FileExistsError(f"Refusing to overwrite {config.output.uri}")
    requested = directory if directory is not None else config.output.work_directory
    if requested is not None and requested.resolve().is_relative_to(
        hdf5_entry_path(config.output.uri)
    ):
        raise ValueError("The work directory must be outside the new IMAS output entry")
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    elif config.output.work_directory is not None:
        base = config.output.work_directory
        base.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="run-", dir=base))
    else:
        directory = create_temporary_workspace(config.output.uri)
    config = replace(config, output=replace(config.output, work_directory=directory))
    save_config(config, directory / "config.toml", overwrite=False)
    write_json(
        directory / "run.json",
        dict(
            kind="validation" if validation else "production",
            components=components,
            geometry=geometry,
        ),
    )
    (directory / "logs").mkdir()
    return config, directory


def load_run(directory: Path, *, validation: bool = False) -> dict:
    """Read a complete run manifest; validation artifacts cannot become production results."""
    manifest = json.loads((directory / "run.json").read_text())
    expected = "validation" if validation else "production"
    if manifest.get("kind") != expected:
        raise ValueError(f"Expected a {expected} run in {directory}")
    components = manifest.get("components", [])
    if not components or len(components) != len(set(components)):
        raise ValueError("Invalid component list in run manifest")
    return manifest


def create_temporary_workspace(output_uri: str) -> Path:
    """Create a unique shared workspace that survives the submitting process."""
    parent = hdf5_entry_path(output_uri).parent
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=_TEMPORARY_PREFIX, dir=parent))
    (directory / _TEMPORARY_MARKER).write_text(str(directory.resolve()))
    return directory


def remove_temporary_workspace(directory: Path, output_uri: str) -> bool:
    """Remove only an automatically created workspace, after successful result storage."""
    directory = directory.resolve()
    marker = directory / _TEMPORARY_MARKER
    if (
        not directory.name.startswith(_TEMPORARY_PREFIX)
        or not marker.is_file()
        or marker.read_text() != str(directory)
    ):
        return False
    if hdf5_entry_path(output_uri).is_relative_to(directory):
        # A task override may deliberately place the final entry in this workspace.
        return False
    shutil.rmtree(directory)
    return True
