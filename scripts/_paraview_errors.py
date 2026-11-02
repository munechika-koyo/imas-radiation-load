"""Display every saved component VTK with absolute or relative Monte Carlo errors."""

from __future__ import annotations

import json
import os
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from paraview import servermanager
from paraview import simple as pv
from vtkmodules.util.numpy_support import vtk_to_numpy

from _configuration import hdf5_entry_path
from _paraview_scene import load_state_tree
from view_results import CONFIGS_DIRECTORY, load_render_config

RELATIVE_ERROR = (
    "numpy.divide(PowerDensityError, numpy.abs(PowerDensity), "
    "out=numpy.full(PowerDensity.shape, numpy.nan), where=PowerDensity != 0, "
    "dtype=numpy.float64) * 100"
)


def component_pipeline(files: list[Path]) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Use independent readers so separate parts cannot become a VTK time series."""
    inputs: list[Any] = []
    statistics: dict[str, dict[str, Any]] = {}
    for path in files:
        pv.SetActiveSource(None)
        source = pv.LegacyVTKReader(registrationName=path.stem, FileNames=[str(path)])
        source.UpdatePipeline()
        for field in ("PowerDensity", "PowerDensityError"):
            if field not in source.CellData.keys():
                raise ValueError(f"{path}: missing cell array {field}")
        relative = pv.PythonCalculator(registrationName=f"{path.stem} error [%]", Input=source)
        relative.ArrayAssociation = "Cell Data"
        relative.ArrayName = "RelativeErrorPercent"
        relative.Expression = RELATIVE_ERROR
        relative.CopyArrays = 1
        relative.UpdatePipeline()
        data = servermanager.Fetch(relative)
        mean = vtk_to_numpy(data.GetCellData().GetArray("PowerDensity"))
        error = vtk_to_numpy(data.GetCellData().GetArray("PowerDensityError"))
        percent = vtk_to_numpy(data.GetCellData().GetArray("RelativeErrorPercent"))
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(error).all()
            or (mean < 0).any()
            or (error < 0).any()
        ):
            raise ValueError(
                f"{path}: power density and standard errors must be finite and nonnegative"
            )
        defined = np.isfinite(percent)
        if not np.array_equal(defined, mean != 0):
            raise ValueError(f"{path}: invalid relative standard errors")
        statistics[path.stem] = dict(
            file=str(path),
            cells=int(mean.size),
            undefined_relative_error_cells=int((~defined).sum()),
            absolute_error_range=[float(error.min()), float(error.max())],
            relative_error_range=[float(percent[defined].min()), float(percent[defined].max())]
            if defined.any()
            else None,
        )
        inputs.append(relative)
    group = pv.GroupDatasets(registrationName="All components", Input=inputs)
    group.BlockNames = [path.stem for path in files]
    group.UpdatePipeline()
    return group, statistics


def remove_component_inputs(state: ET.Element, proxy_id: int) -> None:
    """Remove saved upstream readers so a GUI-saved template can bind a different run."""
    proxies = {proxy.attrib["id"]: proxy for proxy in state.findall("Proxy")}
    removed: set[str] = set()

    def visit(current: str) -> None:
        if current in removed:
            return
        removed.add(current)
        proxy = proxies.get(current)
        if proxy is not None:
            for reference in proxy.findall("./Property[@name='Input']/Proxy"):
                visit(reference.attrib["value"])

    root = str(proxy_id)
    visit(root)
    for current in removed:
        if current in proxies:
            state.remove(proxies[current])
    upstream = removed - {root}
    for parent in state.iter():
        for child in list(parent):
            if (child.tag == "Proxy" and child.get("value") in upstream) or (
                child.tag == "Item" and child.get("id") in upstream
            ):
                parent.remove(child)
        if parent.tag == "Property" and parent.find("Proxy") is not None:
            parent.set("number_of_elements", str(len(parent.findall("Proxy"))))


def bind_components(template: Path, group: Any) -> Any:
    """Bind a variable number of component readers to a common scene template."""
    tree = ET.parse(template)
    state = tree.getroot().find("ServerManagerState")
    if state is None:
        raise ValueError(f"Not a ParaView state: {template}")
    item = state.find("./ProxyCollection[@name='sources']/Item[@name='All components']")
    if item is None:
        raise ValueError("Error template must contain a source named 'All components'")
    proxy_id = int(item.attrib["id"])
    remove_component_inputs(state, proxy_id)
    return load_state_tree(tree, {proxy_id: group})


def build() -> dict[str, Any]:
    start = time.monotonic()
    config = load_render_config(Path(os.environ["RADLOAD_VIEW_CONFIG"]))
    directory = hdf5_entry_path(config.output.uri)
    manifest = json.loads((directory / "rendering.json").read_text())
    expected = set(manifest["components"])
    files = sorted((directory / "vtk").glob("*.vtk"))
    actual = {path.stem for path in files}
    if not expected or actual != expected:
        raise ValueError(
            f"VTK files must match the saved components; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    quantity = os.environ["RADLOAD_ERROR_QUANTITY"]
    if quantity not in {"absolute", "relative"}:
        raise ValueError("Use the relative/absolute quantities defined in the view-errors task")
    pv._DisableFirstRenderCameraReset()
    if not pv.ImportPresets(str(CONFIGS_DIRECTORY / "colormaps/ultraplot-fire.json")):
        raise RuntimeError("Could not import the bundled color presets")
    group, statistics = component_pipeline(files)
    if group.GetDataInformation().GetNumberOfCells() != sum(
        item["cells"] for item in statistics.values()
    ):
        raise ValueError("The combined VTK scene does not contain every component face")
    view = bind_components(Path(os.environ["RADLOAD_VIEW_TEMPLATE"]), group)
    selected: Any = None
    for name in ("absolute", "relative"):
        source = pv.FindSource(f"{name.capitalize()} error")
        if source is None:
            raise ValueError(f"Error template must contain {name.capitalize()!r} error")
        source.UpdatePipeline()
        display = pv.GetDisplayProperties(source, view)
        display.Visibility = int(name == quantity)
        display.SetScalarBarVisibility(view, name == quantity)
        caption = pv.FindSource(f"{name.capitalize()} error interpretation")
        if caption is not None:
            pv.GetDisplayProperties(caption, view).Visibility = int(name == quantity)
        if name == quantity:
            selected = source
            if display.LookupTable.AutomaticRescaleRangeMode != "Never":
                display.RescaleTransferFunctionToDataRange(False, True)
    pv.SetActiveView(view)
    pv.SetActiveSource(selected)
    mode = os.environ["RADLOAD_VIEW_MODE"]
    export_size = tuple(view.ViewSize)
    if mode == "export":
        pv.GetLayout(view).SetSize(*export_size)
    pv.Render(view)
    report = dict(
        status="ready",
        quantity=quantity,
        uncertainty=manifest["uncertainty"],
        relative_error="100 * standard_error / abs(power_density); NaN at zero load",
        config=str(config.source),
        vtk_directory=str(directory / "vtk"),
        components=statistics,
        component_count=len(files),
        cells=group.GetDataInformation().GetNumberOfCells(),
        undefined_relative_error_cells=sum(
            item["undefined_relative_error_cells"] for item in statistics.values()
        ),
        sampling=manifest,
        view=dict(
            size=list(view.ViewSize),
            render_window_size=list(view.GetClientSideObject().GetRenderWindow().GetSize()),
        ),
        elapsed_seconds=time.monotonic() - start,
    )
    if destination := os.environ.get("RADLOAD_VIEW_OUTPUT"):
        path = Path(destination)
        pv.SaveState(str(path / "scene.pvsm"))
        options = dict(ImageResolution=export_size) if mode == "export" else {}
        pv.SaveScreenshot(str(path / "scene.png"), view, **options)
        (path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if mode == "gui":
            pv.Render(view)
    print(
        f"Ready: {quantity} 1σ errors for {len(files)} components ({report['cells']} faces)",
        flush=True,
    )
    return report


if __name__ == "__main__":
    try:
        build()
    except Exception as error:
        if destination := os.environ.get("RADLOAD_VIEW_OUTPUT"):
            (Path(destination) / "report.json").write_text(
                json.dumps(
                    dict(status="failed", error=str(error), traceback=traceback.format_exc()),
                    indent=2,
                )
                + "\n"
            )
        raise
