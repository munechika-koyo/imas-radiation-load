"""Total-power observers and intermediate results, shared with validation."""

from pathlib import Path

import numpy as np
from raysect.core.workflow import MulticoreEngine, SerialEngine
from raysect.optical import World
from raysect.optical.observer import MeshCamera, MonoAdaptiveSampler1D, PowerPipeline1D
from raysect.primitive import Mesh, export_vtk

from _configuration import Config
from _workspace import write_json


def create_camera(
    world: World, mesh: Mesh, workers: int, config: Config
) -> tuple[MeshCamera, PowerPipeline1D]:
    """Use one arbitrary spectral bin: RadiationFunction normalises its integral to total power."""
    pipeline = PowerPipeline1D(accumulate=True)
    sampler = MonoAdaptiveSampler1D(
        pipeline,
        fraction=config.render.adaptive_fraction,
        ratio=config.render.adaptive_ratio,
        min_samples=config.render.min_samples,
        cutoff=config.render.adaptive_cutoff,
    )
    camera = MeshCamera(
        mesh,
        name=f"observer_{mesh.name}",
        surface_offset=config.render.surface_offset,
        pipelines=[pipeline],
        frame_sampler=sampler,
        parent=world,
        min_wavelength=1.0,
        max_wavelength=2.0,
        spectral_bins=1,
        spectral_rays=1,
        pixel_samples=config.render.pixel_samples,
        quiet=True,
    )
    camera.render_engine = SerialEngine() if workers == 1 else MulticoreEngine(processes=workers)
    areas = np.asarray(camera.collection_areas)
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        camera.parent = None
        raise ValueError(f"{mesh.name}: all triangle areas must be finite and positive")
    return camera, pipeline


def save_results(
    directory: Path,
    mesh: Mesh,
    density: np.ndarray,
    error: np.ndarray,
    samples: np.ndarray,
    *,
    complete: bool,
    render_pass: int,
) -> None:
    """Write the same per-face intermediate schema for real and validation data."""
    temporary = directory / "power_density.tmp.npz"
    np.savez_compressed(
        temporary,
        power_density=density,
        power_density_error=error,
        samples=samples,
    )
    temporary.replace(directory / "power_density.npz")
    export_vtk(
        mesh,
        str(directory / "power_density.vtk"),
        triangle_data={"PowerDensity": density, "PowerDensityError": error},
    )
    write_json(directory / "status.json", dict(complete=complete, render_pass=render_pass))


def render_mesh(world: World, mesh: Mesh, directory: Path, workers: int, config: Config) -> bool:
    """Accumulate power until every face satisfies the configured adaptive sampler."""
    camera, pipeline = create_camera(world, mesh, workers, config)
    complete = False
    try:
        areas = np.asarray(camera.collection_areas)
        for render_pass in range(1, config.render.max_passes + 1):
            camera.observe()
            complete = not camera.frame_sampler.generate_tasks(len(areas))
            if complete and (
                not np.isfinite(pipeline.frame.mean).all()
                or not np.isfinite(pipeline.frame.errors()).all()
                or np.any(pipeline.frame.mean < 0)
            ):
                raise ValueError(f"{mesh.name}: converged frame contains invalid powers or errors")
            print(f"{mesh.name}: pass {render_pass}; converged={complete}", flush=True)
            if (
                complete
                or render_pass == config.render.max_passes
                or render_pass % config.render.save_every == 0
            ):
                save_results(
                    directory,
                    mesh,
                    np.asarray(pipeline.frame.mean) / areas,
                    np.asarray(pipeline.frame.errors()) / areas,
                    np.asarray(pipeline.frame.samples),
                    complete=complete,
                    render_pass=render_pass,
                )
            if complete:
                break
    finally:
        camera.parent = None
    return complete
