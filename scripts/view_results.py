"""Launch a shared ParaView state using the rendering configuration through Pixi."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from _configuration import Config, hdf5_entry_path, load_config, require_runtime
from _task_inputs import optional_path, task_inputs

CONFIGS_DIRECTORY = Path(__file__).resolve().parent.parent / "configs"


def load_render_config(path: Path) -> Config:
    """Use the calculation's sources and final output without separate view settings."""
    config = load_config(path)
    require_runtime(config)
    if not hdf5_entry_path(config.output.uri).is_dir():
        raise FileNotFoundError(f"Rendering output does not exist: {config.output.uri}")
    return config


def main() -> int:
    values = task_inputs()
    config = Path(values["RADLOAD_VIEW_CONFIG"]).expanduser().resolve()
    load_render_config(config)
    template = Path(values["RADLOAD_VIEW_TEMPLATE"]).expanduser().resolve()
    if not template.is_file():
        raise FileNotFoundError(f"ParaView state template does not exist: {template}")
    mode = values["RADLOAD_VIEW_MODE"]
    if mode not in {"gui", "export"}:
        raise ValueError("Use the gui/export modes defined in the view-results Pixi task")
    program = "paraview" if mode == "gui" else "pvpython"
    executable = shutil.which(program)
    if executable is None:
        raise FileNotFoundError(f"{program} is missing; use the visualize Pixi environment")
    executable = str(Path(executable).resolve())
    output = optional_path(values["RADLOAD_VIEW_OUTPUT"])
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    elif mode == "export":
        output = Path(tempfile.mkdtemp(prefix="iter-radiation-view-"))
    scripts = Path(__file__).resolve().parent
    env = dict(os.environ)
    env["RADLOAD_VIEW_CONFIG"] = str(config)
    env["RADLOAD_VIEW_TEMPLATE"] = str(template)
    env["RADLOAD_VIEW_OUTPUT"] = str(output) if output else ""
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(scripts), env.get("PYTHONPATH"))))
    args = [executable]
    args += ["--script"] if mode == "gui" else ["--force-offscreen-rendering"]
    scene = (
        "_paraview_errors.py"
        if values.get("RADLOAD_VIEW_KIND") == "errors"
        else "_paraview_scene.py"
    )
    args += [str(scripts / scene)]
    print(f"Rendering configuration: {config}", flush=True)
    print(f"ParaView state template: {template}", flush=True)
    if output:
        print(f"Visualization output: {output}", flush=True)
    os.execve(executable, args, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
