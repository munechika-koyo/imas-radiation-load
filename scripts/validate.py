"""Exercise rendering setup and result I/O without a production calculation."""

import shutil
import tempfile
import traceback
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import numpy as np
from raysect.core import print_scenegraph
from raysect.optical.observer import MeshCamera, PowerPipeline1D
from raysect.primitive import Mesh

from _configuration import hdf5_entry_path, load_config, require_runtime, save_config
from _power_render import create_camera as create_power_camera
from _power_render import save_results
from _probe import SelectedFaces, run_probe
from _runtime import build_runtime, write_component_inputs
from _task_inputs import optional_path, task_inputs
from _wall_io import merge_results
from _workspace import create_run, write_json


def main() -> int:
    values = task_inputs()
    budget = int(values["RADLOAD_PROBE_SAMPLES"])
    if budget < 0:
        raise ValueError("probe_samples must be nonnegative")
    report_directory = optional_path(values["RADLOAD_REPORT_DIR"])
    if report_directory is None:
        report_directory = Path(tempfile.mkdtemp(prefix="iter-radiation-validation-"))
    else:
        report_directory.mkdir(parents=True, exist_ok=False)
    report: dict = dict(status="running", stages=[])
    stage = "configuration"
    print(f"Validation report: {report_directory}", flush=True)

    def passed(name: str) -> None:
        report["stages"].append(dict(name=name, status="passed"))
        write_json(report_directory / "validation.json", report)
        print(f"PASS: {name}", flush=True)

    try:
        config = load_config(Path(values["RADLOAD_CONFIG"]))
        require_runtime(config)
        if hdf5_entry_path(config.output.uri).exists():
            raise FileExistsError(f"Refusing to overwrite {config.output.uri}")
        save_config(config, report_directory / "config.toml", overwrite=False)
        passed(stage)

        stage = "input loading and scene construction"
        world, meshes, geometry, source = build_runtime(config)
        write_json(report_directory / "inputs.json", dict(geometry=geometry, source=source))
        passed(stage)

        def camera_factory(mesh: Mesh) -> tuple[MeshCamera, PowerPipeline1D]:
            return create_power_camera(world, mesh, 1, config)

        stage = "observers and scenegraph"
        scenegraph = report_directory / "scenegraph.txt"
        with scenegraph.open("w") as stream:
            stream.write("Complete scene with one representative observer.\n")
            for index, (name, mesh) in enumerate(meshes.items()):
                camera, _ = camera_factory(mesh)
                try:
                    camera.frame_sampler = SelectedFaces(np.array([], dtype=int))
                    camera.observe()  # Initialise the actual pipelines without tracing any rays.
                    if index == 0:
                        with redirect_stdout(stream):
                            print_scenegraph(world)
                    stream.write(f"\nObserver {name}: {len(camera.collection_areas)} faces\n")
                finally:
                    camera.parent = None
        print(f"Scenegraph saved to: {scenegraph}", flush=True)
        passed(stage)

        stage = "probe"
        report["probe"] = run_probe(meshes, budget, camera_factory)
        if budget:
            passed(stage)
        else:
            report["stages"].append(dict(name=stage, status="skipped", reason="probe_samples=0"))
            print("SKIP: probe (probe_samples=0)", flush=True)

        stage = "intermediate result write/read, merge and HDF5 round trip"
        # Use a sibling of the real entry so this also checks its filesystem/backend.
        parent = hdf5_entry_path(config.output.uri).parent
        parent.mkdir(parents=True, exist_ok=True)
        sandbox = Path(tempfile.mkdtemp(prefix=".iter-radiation-validation-", dir=parent))
        report["sandbox"] = str(sandbox)
        scratch_config = replace(
            config,
            output=replace(
                config.output, uri=f"imas:hdf5?path={sandbox / 'wall'}", work_directory=None
            ),
        )
        scratch_config, run = create_run(
            scratch_config, geometry, list(meshes), directory=sandbox / "run", validation=True
        )
        for index, (name, mesh) in enumerate(meshes.items()):
            directory = run / name
            write_component_inputs(scratch_config, geometry, source, name, directory)
            count = geometry["parts"][name]["triangle_count"]
            # Each component and face has a distinct exactly representable value.
            density = (index + 1) * (
                max(p["triangle_count"] for p in geometry["parts"].values()) + 1
            ) + np.arange(count, dtype=float)
            error = density / 1024.0
            save_results(
                directory, mesh, density, error, np.full(count, 2), complete=True, render_pass=0
            )
            with np.load(directory / "power_density.npz") as data:
                if not np.array_equal(data["power_density"], density) or not np.array_equal(
                    data["power_density_error"], error
                ):
                    raise ValueError(f"{name}: intermediate result round trip failed")
        merge_results(scratch_config, run, validation=True)
        shutil.rmtree(sandbox)
        report["sandbox_removed"] = True
        passed(stage)
        report["status"] = "passed"
        write_json(report_directory / "validation.json", report)
        print(f"Validation passed. Report: {report_directory / 'validation.json'}", flush=True)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["stages"].append(dict(name=stage, status="failed", error=str(error)))
        (report_directory / "error.txt").write_text(traceback.format_exc())
        write_json(report_directory / "validation.json", report)
        print(f"FAIL: {stage}: {error}\nDetails: {report_directory / 'error.txt'}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
