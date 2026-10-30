"""IMAS wall geometry, direct component loading and single-writer result assembly."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.wall import load_wall_3d
from cherab.imas.wall import load_wall_mesh
from imas import DBEntry, IDSFactory, identifiers
from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS
from imas.ids_toplevel import IDSToplevel
from numpy.typing import NDArray
from raysect.optical.material import AbsorbingSurface
from raysect.primitive import Mesh

from _configuration import (
    Config,
    GeometryObjectConfig,
    hdf5_entry_path,
    load_config,
    require_runtime,
)
from _workspace import load_run, write_json


def file_digest(path: Path) -> str:
    """Hash a local input without loading it all into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def geometry_digest(vertices: NDArray[np.float64], triangles: NDArray[np.int32]) -> str:
    """Hash ordered triangle coordinates, preserving face winding and face order."""
    return hashlib.sha256(vertices[triangles].astype("<f8").tobytes()).hexdigest()


def wall_identity(config: Config) -> dict:
    target = config.geometry.target
    path = Path(target.uri)
    return dict(
        uri=target.uri,
        occurrence=target.occurrence,
        description_index=target.description_index,
        time=target.time,
        # Record the input content for result provenance.
        source_sha256=file_digest(path) if path.is_file() else None,
    )


def load_wall_parts(config: Config, geometry: dict) -> dict[str, Mesh]:
    """Load component meshes directly from the configured IMAS wall entry."""
    target = config.geometry.target
    meshes = load_wall_mesh(
        target.uri,
        time=target.time,
        occurrence=target.occurrence,
        desc_ggd_index=target.description_index,
        subsets=list(geometry["parts"]),
    )
    parts: dict[str, Mesh] = {}
    for name, part in geometry["parts"].items():
        mesh = meshes[f"{geometry['grid_name']}.{name}.none"]
        vertices = np.asarray(mesh.data.vertices, dtype=np.float64)
        triangles = np.asarray(mesh.data.triangles, dtype=np.int32)[:, :3]
        if (
            len(triangles) != part["triangle_count"]
            or geometry_digest(vertices, triangles) != part["mesh_digest"]
        ):
            raise ValueError(f"{name}: loaded mesh differs from the wall subset geometry")
        mesh.name = name
        mesh.material = AbsorbingSurface()
        parts[name] = mesh
    return parts


def load_non_target(source: GeometryObjectConfig) -> dict[str, Mesh]:
    """Load all wall faces as absorbers, including faces outside named subsets."""
    with DBEntry(source.uri, "r") as entry:
        wall = get_ids_time_slice(entry, "wall", time=source.time, occurrence=source.occurrence)
        description = wall.description_ggd[source.description_index]
        if len(description.grid_ggd) != 1:
            raise ValueError(f"{source.uri}: non-target geometry requires one wall grid")
        grid = description.grid_ggd[0]
        if len(grid.space) != 1 or [
            int(getattr(c, "index", c)) for c in grid.space[0].coordinates_type
        ] != [
            int(getattr(identifiers.coordinate_identifier, axis).index) for axis in ("x", "y", "z")
        ]:
            raise ValueError(
                f"{source.uri}: non-target geometry requires one Cartesian x/y/z space"
            )
        components = load_wall_3d(description)
    meshes = {
        name: Mesh(
            component["vertices"],
            component["triangles"],
            closed=False,
            material=AbsorbingSurface(),
            name=name,
        )
        for name, component in components.items()
        # The loader also returns an empty remainder when subsets cover every face.
        if len(component["triangles"])
    }
    if not meshes:
        raise ValueError(f"{source.uri}: non-target geometry contains no wall faces")
    return meshes


def _write_wall(wall: IDSToplevel, output_uri: str, *, occurrence: int = 0) -> None:
    """Create a new HDF5 entry and clean up only this entry if writing fails."""
    wall.validate()
    output = hdf5_entry_path(output_uri)
    output.mkdir(parents=True, exist_ok=False)
    try:
        with DBEntry(output_uri, "x", dd_version=wall._version) as entry:
            entry.put(wall, occurrence=occurrence)
    except BaseException:
        shutil.rmtree(output)
        raise


def wall_geometry(config: Config) -> dict:
    """Read component identities and ordered geometry from the source wall IDS."""
    target = config.geometry.target
    with DBEntry(target.uri, "r") as entry:
        wall = entry.get("wall", occurrence=target.occurrence, autoconvert=False, lazy=True)
        description = wall.description_ggd[target.description_index]
        if len(description.grid_ggd) != 1 or len(description.material):
            raise ValueError(
                "This workflow requires one triangular grid and no material-split subsets"
            )
        grid = description.grid_ggd[0]
        if len(grid.space) != 1 or [int(c.index) for c in grid.space[0].coordinates_type] != [
            int(getattr(identifiers.coordinate_identifier, axis).index) for axis in ("x", "y", "z")
        ]:
            raise ValueError("The wall grid must have one Cartesian x/y/z space")
        grid_name = str(grid.identifier.name)
        parts = {
            str(s.identifier.name): dict(
                subset_index=int(s.identifier.index), triangle_count=len(s.element)
            )
            for s in grid.grid_subset
            if int(s.dimension) == 3
        }
        if len(parts) != len(grid.grid_subset):
            raise ValueError(
                "Use a grid with face subsets only; non-face subset positions are not supported by this wall export yet"
            )
        if not parts or any(part["triangle_count"] == 0 for part in parts.values()):
            raise ValueError("Explicit nonempty triangle subsets are required")
        for name in parts:
            if (
                not name.strip()
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                or "\0" in name
            ):
                raise ValueError(
                    f"Wall component name cannot be used as a result directory: {name!r}"
                )
        if len({part["subset_index"] for part in parts.values()}) != len(parts):
            raise ValueError("Wall component grid_subset identifiers must be unique")
        components = load_wall_3d(description, subsets=list(parts))
        for name, part in parts.items():
            component = components[f"{grid_name}.{name}.none"]
            if len(component["triangles"]) != part["triangle_count"]:
                raise ValueError(
                    f"{name}: each wall subset element must be a triangle; polygon triangulation changes the result mapping"
                )
            part["mesh_digest"] = geometry_digest(component["vertices"], component["triangles"])
        return dict(
            source=wall_identity(config),
            dd_version=wall._version,
            grid_index=1,
            grid_name=grid_name,
            parts=parts,
        )


def merge_results(config: Config, run_directory: Path, *, validation: bool = False) -> str:
    """Assemble finished estimates, uncertainties and convergence records using one writer."""
    require_runtime(config)
    geometry = wall_geometry(config)
    manifest = load_run(run_directory, validation=validation)
    if manifest["geometry"] != geometry:
        raise ValueError("Wall geometry differs from the frozen run manifest")
    targets = manifest["components"]
    unexpected = {
        path.name
        for path in run_directory.iterdir()
        if path.is_dir() and (path / "identity.json").is_file()
    } - set(targets)
    if unexpected:
        raise ValueError(f"Unexpected component results: {sorted(unexpected)}")
    values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sampling: dict = dict(
        max_passes=config.render.max_passes,
        adaptive_cutoff=config.render.adaptive_cutoff,
        uncertainty="1-sigma Monte Carlo standard error of the mean [W m-2]",
        components={},
    )
    source_info: dict | None = None
    for name in targets:
        directory = run_directory / name
        identity = json.loads((directory / "identity.json").read_text())
        if name not in geometry["parts"]:
            raise ValueError(f"{name}: component does not exist in the source wall IDS")
        if identity != dict(source=geometry["source"], part=name, **geometry["parts"][name]):
            raise ValueError(f"{name}: wall identity/triangle order does not match the result")
        status = json.loads((directory / "status.json").read_text())
        render_pass = status["render_pass"]
        # Older workers used complete solely for convergence and exited 2 at the cap.
        legacy = "converged" not in status
        converged = status.get("converged", status["complete"])
        user_stopped = status.get("stop_reason") == "user_stop" and status["complete"]
        finished = status["complete"] or (legacy and render_pass == config.render.max_passes)
        if (
            not finished
            or not isinstance(converged, bool)
            or not isinstance(render_pass, int)
            or render_pass < (0 if validation else 1)
            or render_pass > config.render.max_passes
            or (not converged and not user_stopped and render_pass != config.render.max_passes)
        ):
            raise ValueError(
                f"{name}: results must be converged, pass-limited, or explicitly stopped by the user"
            )
        if not (directory / "power_density.vtk").is_file():
            raise FileNotFoundError(f"{name}: the VTK result with standard errors is missing")
        recorded = load_config(directory / "config.toml")
        if (
            recorded.plasma != config.plasma
            or recorded.atomic != config.atomic
            or recorded.render != config.render
            or recorded.emitter != config.emitter
            or recorded.radiation != config.radiation
            or recorded.geometry != config.geometry
        ):
            raise ValueError(
                f"{name}: emitter/plasma/atomic/render/geometry configuration differs from the collection configuration"
            )
        component_source = json.loads((directory / "source.json").read_text())
        if component_source.get("kind") != config.emitter.kind:
            raise ValueError(f"{name}: source does not match the configured emitter")
        if source_info is not None and component_source != source_info:
            raise ValueError(f"{name}: radiation inputs differ from the other components")
        source_info = component_source
        with np.load(directory / "power_density.npz") as data:
            density, error = data["power_density"].copy(), data["power_density_error"].copy()
            samples = data["samples"].copy()
        if (
            density.shape != (geometry["parts"][name]["triangle_count"],)
            or error.shape != density.shape
            or samples.shape != density.shape
        ):
            raise ValueError(f"{name}: wrong number of triangle values")
        if (
            not np.isfinite(density).all()
            or not np.isfinite(error).all()
            or np.any(density < 0)
            or np.any(error < 0)
            or not np.issubdtype(samples.dtype, np.integer)
            or np.any(samples < 1)
        ):
            raise ValueError(f"{name}: results must be finite and nonnegative")
        values[name] = density, error
        sampling["components"][name] = dict(
            converged=converged,
            stop_reason="converged" if converged else "user_stop" if user_stopped else "max_passes",
            render_pass=render_pass,
            samples_min=int(samples.min()),
            samples_max=int(samples.max()),
        )
    assert source_info is not None
    time = float(source_info["time"])
    if not np.isfinite(time):
        raise ValueError("Result time must be finite")
    target = config.geometry.target
    with DBEntry(target.uri, "r") as entry:
        source_wall = entry.get("wall", occurrence=target.occurrence, autoconvert=False)
    source_description = source_wall.description_ggd[target.description_index]
    # Keep only the selected geometry; never re-date unrelated input physics arrays.
    wall = IDSFactory(source_wall._version).wall()
    wall.description_ggd.resize(1)
    description = wall.description_ggd[0]
    description.type = source_description.type
    description.grid_ggd.resize(1)
    description.grid_ggd[0] = source_description.grid_ggd[0]
    wall.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    wall.ids_properties.creation_date = datetime.now(UTC).isoformat()
    wall.time = [time]
    for grid in description.grid_ggd:
        grid.time = time
    description.ggd.resize(1)
    description.ggd[0].time = time
    # IMAS-ParaView 2.4 also uses positional lookup for this wall quantity.
    description.ggd[0].power_density.resize(len(geometry["parts"]))
    for quantity, name in zip(description.ggd[0].power_density, geometry["parts"], strict=True):
        quantity.grid_index = 1
        quantity.grid_subset_index = geometry["parts"][name]["subset_index"]
        if name not in values:
            continue
        density, error = values[name]
        quantity.values = density
        quantity.values_error_upper = error
        quantity.values_error_lower = error
    emission = "Total radiation from radiation IDS. "
    wall.ids_properties.comment = (
        ("VALIDATION DATA; not a physical result. " if validation else "")
        + emission
        + "Absorbing non-emitting surfaces: incident radiation equals net radiation; "
        "particle heat flux, wall thermal emission and reflection are excluded. Units W m-2. "
        "Symmetric values_error_upper/lower are 1-sigma Monte Carlo standard errors. "
        "Estimates are saved at max_passes or an explicit user stop, even before convergence. "
        f"Sampling: {json.dumps(sampling, sort_keys=True)}. "
        "Only measured subsets have values; unmeasured subset entries have empty values. "
        f"Reference wall geometry; {target.num_segments} toroidal placements were used for occlusion. "
        f"Wall source: {json.dumps(geometry['source'], sort_keys=True)}. "
        f"Emission source: {json.dumps(source_info, sort_keys=True)}"
    )
    wall.code.name = "iter-radiation-load"
    wall.code.version = "0.1.0"
    _write_wall(wall, config.output.uri, occurrence=config.output.occurrence)
    verify_wall_results(config, geometry, values, time)
    output = hdf5_entry_path(config.output.uri)
    vtk_directory = output / "vtk"
    vtk_directory.mkdir()
    for name in targets:
        shutil.copyfile(run_directory / name / "power_density.vtk", vtk_directory / f"{name}.vtk")
    write_json(output / "rendering.json", sampling)
    return config.output.uri


def verify_wall_results(
    config: Config,
    geometry: dict,
    values: dict[str, tuple[np.ndarray, np.ndarray]],
    time: float,
) -> None:
    """Read the saved HDF5 entry and verify face values, errors and empty unmeasured subsets."""
    with DBEntry(config.output.uri, "r") as entry:
        wall = entry.get("wall", occurrence=config.output.occurrence, autoconvert=False)
    if not np.array_equal(np.asarray(wall.time), [time]):
        raise ValueError("Saved wall time does not match the calculated result time")
    description = wall.description_ggd[0]
    grid = description.grid_ggd[0]
    quantities = description.ggd[0].power_density
    if len(grid.grid_subset) != len(geometry["parts"]) or len(quantities) != len(geometry["parts"]):
        raise ValueError("Saved wall has an inconsistent number of component subsets")
    for subset, quantity, (name, part) in zip(
        grid.grid_subset, quantities, geometry["parts"].items(), strict=True
    ):
        if (
            str(subset.identifier.name) != name
            or int(subset.identifier.index) != part["subset_index"]
            or int(quantity.grid_index) != 1
            or int(quantity.grid_subset_index) != part["subset_index"]
        ):
            raise ValueError(f"{name}: saved grid_subset mapping differs from the input")
        expected = values.get(name)
        if expected is None:
            if (
                len(quantity.values)
                or len(quantity.values_error_upper)
                or len(quantity.values_error_lower)
            ):
                raise ValueError(f"{name}: an unmeasured component must have empty values")
            continue
        density, error = expected
        if not (
            np.array_equal(np.asarray(quantity.values), density)
            and np.array_equal(np.asarray(quantity.values_error_upper), error)
            and np.array_equal(np.asarray(quantity.values_error_lower), error)
        ):
            raise ValueError(
                f"{name}: saved face values or errors differ from the calculated result"
            )
