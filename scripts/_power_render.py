"""Total-power observers and intermediate results, shared with validation."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from raysect.core.workflow import MulticoreEngine, SerialEngine
from raysect.optical import World
from raysect.optical.observer import MeshCamera, MonoAdaptiveSampler1D, PowerPipeline1D
from raysect.primitive import Mesh, export_vtk

from _configuration import Config
from _timing import RenderTiming
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
    converged: bool,
    render_pass: int,
    timing: RenderTiming | None = None,
) -> None:
    """Write the same per-face intermediate schema for real and validation data."""
    save_started = perf_counter()
    if (
        not np.isfinite(density).all()
        or not np.isfinite(error).all()
        or np.any(density < 0)
        or np.any(error < 0)
    ):
        raise ValueError(f"{mesh.name}: powers and standard errors must be finite and nonnegative")
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
    status: dict[str, object] = dict(
        complete=complete,
        converged=converged,
        render_pass=render_pass,
        stop_reason="converged" if converged else "max_passes" if complete else "in_progress",
    )
    if timing is not None:
        timing.save_seconds += perf_counter() - save_started
        status["timing"] = timing.snapshot()
    write_json(directory / "status.json", status)


def render_mesh(
    world: World,
    mesh: Mesh,
    directory: Path,
    workers: int,
    config: Config,
    *,
    timing: RenderTiming | None = None,
) -> bool:
    """Stop on convergence or at max_passes; return whether the sampler converged."""
    timing = timing if timing is not None else RenderTiming(workers)
    camera, pipeline = create_camera(world, mesh, workers, config)
    converged = False
    try:
        areas = np.asarray(camera.collection_areas)
        timing.finish_setup()
        for render_pass in range(1, config.render.max_passes + 1):
            pass_started = perf_counter()
            camera.observe()
            timing.rendering_seconds += perf_counter() - pass_started
            converged = not camera.frame_sampler.generate_tasks(len(areas))
            finished = converged or render_pass == config.render.max_passes
            print(f"{mesh.name}: pass {render_pass}; converged={converged}", flush=True)
            if finished or render_pass % config.render.save_every == 0:
                save_results(
                    directory,
                    mesh,
                    np.asarray(pipeline.frame.mean) / areas,
                    np.asarray(pipeline.frame.errors()) / areas,
                    np.asarray(pipeline.frame.samples),
                    complete=finished,
                    converged=converged,
                    render_pass=render_pass,
                    timing=timing,
                )
            if converged:
                break
    finally:
        camera.parent = None
    status = json.loads((directory / "status.json").read_text())
    status["timing"] = timing.snapshot(finished=True)
    write_json(directory / "status.json", status)
    return converged
