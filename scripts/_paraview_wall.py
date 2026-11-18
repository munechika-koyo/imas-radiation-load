"""Build the shared wall source and validate per-component display arrays."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from paraview import servermanager
from paraview import simple as pv
from vtkmodules.util.numpy_support import vtk_to_numpy


def wall_pipeline(uri: str, occurrence: int) -> tuple[Any, dict[str, dict[str, Any]], float]:
    """Read the IDS on the server so saved ParaView states can reopen the same source."""
    group = pv.ProgrammableSource(registrationName="wall IDS")
    group.OutputDataSetType = "vtkMultiblockDataSet"
    group.Script = (
        "import sys\n"
        f"directory = {str(Path(__file__).resolve().parent)!r}\n"
        "if directory not in sys.path:\n"
        "    sys.path.insert(0, directory)\n"
        "from _wall_data import read_wall_results\n"
        f"data = read_wall_results({uri!r}, occurrence={occurrence!r})\n"
        "self.GetOutputDataObject(0).ShallowCopy(data)\n"
    )
    group.UpdatePipeline()
    blocks = servermanager.Fetch(group)
    if blocks.GetNumberOfBlocks() == 0:
        raise ValueError("Wall IDS reader produced no measured components; see ParaView log")
    statistics: dict[str, dict[str, Any]] = {}
    for index in range(blocks.GetNumberOfBlocks()):
        data = blocks.GetBlock(index)
        # Fetch may prefix composite block names while combining server pieces.
        name = data.GetFieldData().GetAbstractArray("Component").GetValue(0)
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
                f"{name}: power density and standard errors must be finite and nonnegative"
            )
        defined = np.isfinite(percent)
        if not np.array_equal(defined, mean != 0):
            raise ValueError(f"{name}: invalid relative standard errors")
        statistics[name] = dict(
            cells=int(mean.size),
            undefined_relative_error_cells=int((~defined).sum()),
            absolute_error_range=[float(error.min()), float(error.max())],
            relative_error_range=[float(percent[defined].min()), float(percent[defined].max())]
            if defined.any()
            else None,
        )
    time = float(vtk_to_numpy(blocks.GetFieldData().GetArray("Time"))[0])
    return group, statistics, time


def remove_component_inputs(state: ET.Element, proxy_id: str) -> None:
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

    visit(proxy_id)
    for current in removed:
        if current in proxies:
            state.remove(proxies[current])
    upstream = removed - {proxy_id}
    if not upstream:
        return
    for parent in state.iter():
        for child in list(parent):
            if (child.tag == "Proxy" and child.get("value") in upstream) or (
                child.tag == "Item" and child.get("id") in upstream
            ):
                parent.remove(child)
        if parent.tag == "Property" and parent.find("Proxy") is not None:
            parent.set("number_of_elements", str(len(parent.findall("Proxy"))))
