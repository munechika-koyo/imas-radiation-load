"""Create an IMAS HDF5 wall entry once from STL files, removing duplicate faces.

This standalone script uses only external libraries and Python's standard library.
Pixi owns task arguments and supplies them through environment variables.
Duplicate triangles are removed within each component, ignoring vertex winding.
The first occurrence of each triangle is kept in source order.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from imas import DBEntry, IDSFactory, identifiers
from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS
from imas.ids_toplevel import IDSToplevel
from numpy.typing import NDArray
from raysect.primitive import import_stl


@dataclass(frozen=True)
class PartGeometry:
    """One STL component after mandatory coordinate welding and face deduplication."""

    source: Path
    vertices: NDArray[np.float64]
    triangles: NDArray[np.int64]
    removed_triangles: int


def find_stl_files(directory: Path) -> list[Path]:
    """Find component STL files directly inside a directory in stable name order."""
    if not directory.is_dir():
        raise NotADirectoryError(f"STL directory does not exist or is not a directory: {directory}")
    paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".stl"
    )
    if not paths:
        raise ValueError(f"No STL files found in {directory}")
    names: set[str] = set()
    for path in paths:
        name = path.stem
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"{path.name}: component names must use letters, numbers, '_' or '-'")
        if name in names:
            raise ValueError(f"Duplicate component name {name!r}")
        names.add(name)
    return paths


def load_part(path: Path, scaling: float, flip_normals: bool) -> PartGeometry:
    """Read an STL with Raysect and deduplicate faces using their actual coordinates."""
    # CAD/STL inputs can contain zero-area facets. Raysect removes these before
    # computing normals; the surviving faces are then deduplicated below.
    mesh = import_stl(str(path), scaling=scaling, closed=False, tolerant=True)
    vertices = np.asarray(mesh.data.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.data.triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"{path}: vertices must be finite Cartesian triples")
    if triangles.ndim != 2 or triangles.shape[1] < 3 or len(triangles) == 0:
        raise ValueError(f"{path}: at least one triangle is required")
    triangles = triangles[:, :3]
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError(f"{path}: triangle vertex index is out of range")

    # STL facets may use different indices for the same coordinates. Weld first
    # so winding-independent face keys also identify those geometric duplicates.
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    triangles = inverse[triangles]
    keys = np.sort(triangles, axis=1)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(first_indices)
    removed = len(triangles) - len(keep)
    triangles = triangles[keep]
    if flip_normals:
        triangles = triangles[:, ::-1]
    print(
        f"{path.name}: triangles {len(keep) + removed} -> {len(keep)} (removed {removed})",
        flush=True,
    )
    return PartGeometry(path, vertices, triangles, removed)


def build_wall(
    paths: list[Path], scaling: float, flip_normals: bool, dd_version: str
) -> IDSToplevel:
    """Build one shared Cartesian grid and an explicit face subset per component."""
    parts = [load_part(path, scaling, flip_normals) for path in paths]
    vertices, inverse = np.unique(
        np.vstack([part.vertices for part in parts]), axis=0, return_inverse=True
    )
    all_triangles: list[NDArray[np.int64]] = []
    vertex_offset = 0
    for part in parts:
        all_triangles.append(inverse[part.triangles + vertex_offset])
        vertex_offset += len(part.vertices)
    triangles = np.vstack(all_triangles)
    points = vertices[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
    )
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("Triangle areas must be finite and positive")
    if max(len(vertices), len(triangles)) > np.iinfo(np.int32).max:
        raise ValueError("Wall geometry exceeds the supported IMAS index range")
    print(
        f"Building wall IDS: {len(parts)} parts, {len(vertices)} unique vertices, "
        f"{len(triangles)} triangles",
        flush=True,
    )

    wall = IDSFactory(dd_version).wall()
    wall.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    wall.ids_properties.creation_date = datetime.now(UTC).isoformat()
    wall.ids_properties.comment = (
        "Reference wall geometry assembled from STL components; Cartesian metres. "
        "Degenerate triangles discarded by Raysect on STL import. "
        "Exact duplicate vertices welded. Duplicate triangles removed per component, "
        "ignoring winding; first occurrence and relative face order retained. "
        f"STL scaling: {scaling}; STL winding reversed: {flip_normals}. "
        f"Removed triangles: {sum(part.removed_triangles for part in parts)}."
    )
    wall.time = [0.0]
    wall.code.name = "imas-radiation-load"
    wall.code.version = "0.1.0"
    wall.description_ggd.resize(1)
    description = wall.description_ggd[0]
    description.type.index = 1
    description.type.name = "wall with holes"
    description.grid_ggd.resize(1)
    grid = description.grid_ggd[0]
    grid.identifier = identifiers.ggd_identifier.linear
    grid.time = 0.0
    grid.space.resize(1)
    space = grid.space[0]
    space.identifier = identifiers.ggd_space_identifier.primary_standard
    space.geometry_type.index = 0
    space.geometry_type.name = "standard"
    space.coordinates_type.resize(3)
    for coordinate, axis in zip(space.coordinates_type, ("x", "y", "z"), strict=True):
        reference = getattr(identifiers.coordinate_identifier, axis)
        coordinate.index = reference.index
        coordinate.name = axis
    space.objects_per_dimension.resize(3)
    nodes = space.objects_per_dimension[0].object
    nodes.resize(len(vertices))
    for node, xyz in zip(nodes, vertices, strict=True):
        node.geometry = xyz
    faces = space.objects_per_dimension[2].object
    faces.resize(len(triangles))
    for index, (face, triangle, area) in enumerate(zip(faces, triangles, areas, strict=True)):
        face.nodes = np.asarray(triangle + 1, dtype=np.int32)
        face.measure = area
        if index and index % 100000 == 0:
            print(f"  geometry faces: {index}/{len(triangles)}", flush=True)

    grid.grid_subset.resize(len(parts))
    face_offset = 0
    for index, (part, subset) in enumerate(zip(parts, grid.grid_subset, strict=True)):
        subset.identifier.name = part.source.stem
        subset.identifier.index = -(index + 1)
        subset.identifier.description = (
            f"Source STL: {part.source.name}; removed {part.removed_triangles} duplicate triangles"
        )
        subset.dimension = 3
        subset.element.resize(len(part.triangles))
        for face_index, element in enumerate(subset.element):
            element.object.resize(1)
            element.object[0].space = 1
            element.object[0].dimension = 3
            element.object[0].index = face_offset + face_index + 1
        face_offset += len(part.triangles)
        print(f"  subset: {part.source.stem} ({len(part.triangles)} faces)", flush=True)
    return wall


def write_wall(wall: IDSToplevel, output: Path) -> None:
    """Create a new HDF5 entry directory at occurrence zero without overwriting."""
    wall.validate()
    # Claim a new directory first. Never write into or clean up an existing entry.
    output.mkdir(parents=True, exist_ok=False)
    try:
        with DBEntry(f"imas:hdf5?path={output}", "x", dd_version=wall._version) as entry:
            entry.put(wall)
    except BaseException:
        shutil.rmtree(output)
        raise


def main() -> int:
    """Create geometry using arguments supplied exclusively by the Pixi task."""
    if len(sys.argv) != 1:
        raise SystemExit("Pass arguments through the prepare-wall Pixi task.")
    directory = Path(os.environ["RADLOAD_MESH_DIR"]).expanduser().resolve()
    output = Path(os.environ["RADLOAD_OUTPUT"]).expanduser().resolve()
    scaling = float(os.environ["RADLOAD_STL_SCALING"])
    flip_normals = os.environ["RADLOAD_FLIP_NORMALS"] == "true"
    dd_version = os.environ["RADLOAD_DD_VERSION"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("stl_scaling must be finite and positive")
    wall = build_wall(find_stl_files(directory), scaling, flip_normals, dd_version)
    write_wall(wall, output)
    print(
        f"Wall IDS: imas:hdf5?path={output} (occurrence=0, description_ggd[0], time=0)", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
