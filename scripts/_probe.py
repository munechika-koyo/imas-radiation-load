"""Deterministic face selection with an exact total primary-sample budget."""

from collections.abc import Callable
from time import perf_counter

import numpy as np
from raysect.core.workflow import SerialEngine
from raysect.optical.observer import FrameSampler1D, MeshCamera, PowerPipeline1D
from raysect.primitive import Mesh


class SelectedFaces(FrameSampler1D):
    """Request only the selected camera pixels, once per observe call."""

    def __init__(self, indices: np.ndarray) -> None:
        self.indices = indices

    def generate_tasks(self, pixels: int) -> list[tuple[int]]:
        if np.any(self.indices < 0) or np.any(self.indices >= pixels):
            raise ValueError("Probe face index is outside the observer")
        return [(int(index),) for index in self.indices]


def representative_faces(mesh: Mesh, count: int) -> np.ndarray:
    """Spread selected face centres through the component using deterministic farthest points."""
    triangles = np.asarray(mesh.data.triangles)[:, :3]
    count = min(count, len(triangles))
    if count == len(triangles):
        return np.arange(count, dtype=np.int32)
    candidates = np.unique(
        np.linspace(0, len(triangles) - 1, min(len(triangles), max(2048, count)), dtype=np.int32)
    )
    centres = np.asarray(mesh.data.vertices)[triangles[candidates]].mean(axis=1)
    distances = np.full(len(candidates), np.inf)
    selected = []
    index = int(np.argmin(centres[:, 0]))
    for _ in range(count):
        selected.append(index)
        distances = np.minimum(distances, np.square(centres - centres[index]).sum(axis=1))
        distances[selected] = -1
        index = int(np.argmax(distances))
    return candidates[selected]


def run_probe(
    meshes: dict[str, Mesh],
    budget: int,
    camera_factory: Callable[[Mesh], tuple[MeshCamera, PowerPipeline1D]],
) -> dict:
    """Distribute a total budget across components; geometry remains untouched."""
    if budget < 0:
        raise ValueError("probe_samples must be nonnegative")
    started = perf_counter()
    names = list(meshes)
    report: dict = dict(requested_samples=budget, actual_samples=0, components=[], unchecked=[])
    if budget == 0:
        report.update(status="skipped", unchecked=names, elapsed_seconds=0.0)
        return report
    selected = np.linspace(0, len(names) - 1, min(len(names), budget), dtype=int)
    allocation, extra = divmod(budget, len(selected))
    report["unchecked"] = [name for i, name in enumerate(names) if i not in selected]
    for order, component_index in enumerate(selected):
        name = names[int(component_index)]
        component_started = perf_counter()
        samples = allocation + (order < extra)
        # Bound face-selection work too; larger budgets add samples to these faces.
        indices = representative_faces(meshes[name], min(samples, 64))
        camera, pipeline = camera_factory(meshes[name])
        try:
            camera.render_engine = SerialEngine()
            camera.spectral_rays = 1
            full, remainder = divmod(samples, len(indices))
            for subset, count in ((indices, full), (indices[:remainder], 1)):
                if not len(subset) or not count:
                    continue
                camera.pixel_samples = count
                camera.frame_sampler = SelectedFaces(subset)
                camera.observe()
            observed = int(np.asarray(pipeline.frame.samples).sum())
            means = np.asarray(pipeline.frame.mean)[indices]
            if observed != samples or not np.isfinite(means).all() or np.any(means < 0):
                raise ValueError(f"{name}: invalid probe result or sample count")
            report["actual_samples"] += observed
            report["components"].append(
                dict(
                    name=name,
                    triangle_indices=indices.tolist(),
                    samples=observed,
                    nonzero_faces=int(np.count_nonzero(means)),
                    max_power_W=float(means.max()),
                    elapsed_seconds=perf_counter() - component_started,
                )
            )
        finally:
            camera.parent = None
    report["status"] = "passed"
    report["elapsed_seconds"] = perf_counter() - started
    return report
