"""Construct the absorbing geometry shared by rendering and validation."""

from raysect.core.math import rotate_z
from raysect.core.scenegraph import Node
from raysect.optical import World
from raysect.optical.material import AbsorbingSurface
from raysect.primitive import Mesh

from _configuration import Config
from _wall_io import load_non_target, load_wall_parts


def add_reference_segments(world: World, meshes: dict[str, Mesh], num_segments: int) -> None:
    """Group reference geometry separately from its occluding copies."""
    original = Node(parent=world, name="reference_segment")
    for mesh in meshes.values():
        mesh.parent = original
    for segment in range(1, num_segments):
        node = Node(
            parent=world,
            name=f"copy_segment_{segment}",
            transform=rotate_z(segment * 360.0 / num_segments),
        )
        for part, mesh in meshes.items():
            mesh.instance(
                parent=node,
                transform=mesh.transform,
                material=AbsorbingSurface(),
                name=f"{part}_copy_{segment}",
            )


def add_non_target_geometry(
    world: World, meshes: dict[str, Mesh], name: str, num_segments: int
) -> None:
    """Attach non-target meshes and their rotated copies directly to the world."""
    for part, mesh in meshes.items():
        mesh.name = name if len(meshes) == 1 else f"{name}.{part}"
        mesh.parent = world
        for segment in range(1, num_segments):
            mesh.instance(
                parent=world,
                transform=rotate_z(segment * 360.0 / num_segments) * mesh.transform,
                material=AbsorbingSurface(),
                name=f"{mesh.name}_copy_{segment}",
            )


def create_scene(config: Config, geometry: dict) -> tuple[World, dict[str, Mesh]]:
    """Return scene geometry and reference meshes used to construct MeshCamera observers."""
    world = World()
    meshes = load_wall_parts(config, geometry)
    add_reference_segments(world, meshes, config.geometry.target.num_segments)
    for source in config.geometry.non_targets:
        add_non_target_geometry(world, load_non_target(source), source.name, source.num_segments)
    return world, meshes
