"""Shared scene initialisation and per-component execution."""

from pathlib import Path

from raysect.optical import World
from raysect.primitive import Mesh

from _configuration import Config, require_runtime, save_config
from _scene import create_scene
from _source import load_source
from _timing import RenderTiming
from _wall_io import wall_geometry
from _workspace import load_run, write_json


def build_runtime(
    config: Config, geometry: dict | None = None
) -> tuple[World, dict[str, Mesh], dict, dict]:
    """Perform the identical input loading and scene construction for all tasks."""
    require_runtime(config)
    print("Loading target wall geometry...", flush=True)
    geometry = geometry if geometry is not None else wall_geometry(config)
    print("Constructing scene geometry...", flush=True)
    world, meshes = create_scene(config, geometry)
    print("Loading emission source...", flush=True)
    source = load_source(world, config)
    return world, meshes, geometry, source


def write_component_inputs(
    config: Config,
    geometry: dict,
    source: dict,
    component: str,
    directory: Path,
) -> None:
    """Claim a component directory and record its immutable face mapping and inputs."""
    directory.mkdir()
    save_config(config, directory / "config.toml", overwrite=False)
    write_json(
        directory / "identity.json",
        dict(source=geometry["source"], part=component, **geometry["parts"][component]),
    )
    write_json(directory / "source.json", source)


def render_component(config: Config, component: str, directory: Path, workers: int) -> bool:
    """Calculate one planned component and return convergence; the pass limit is a valid stop."""
    timing = RenderTiming(workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    manifest = load_run(directory)
    if component not in manifest["components"]:
        raise ValueError(f"{component!r} is not a component of this run")
    geometry = wall_geometry(config)
    if geometry != manifest["geometry"]:
        raise ValueError("Wall input differs from the frozen run geometry")
    world, meshes, _, source = build_runtime(config, geometry)
    destination = directory / component
    write_component_inputs(config, geometry, source, component, destination)
    print(f"Component: {component}; work directory: {directory}; workers: {workers}", flush=True)
    from _power_render import render_mesh

    return render_mesh(world, meshes[component], destination, workers, config, timing=timing)
