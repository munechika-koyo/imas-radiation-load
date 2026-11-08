"""Export saved camera viewpoints with linear and logarithmic scalar colors."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from paraview import servermanager
from paraview import simple as pv
from vtkmodules.util.numpy_support import vtk_to_numpy


@dataclass(frozen=True)
class Viewpoint:
    name: str
    stem: str
    properties: dict[str, list[float]]


def load_viewpoints(path: Path) -> list[Viewpoint]:
    """Read native ParaView custom-viewpoint files without global GUI settings."""
    root = ET.parse(path).getroot()
    if root.tag != "CustomViewpointsConfiguration":
        raise ValueError(f"Not a ParaView custom-viewpoint file: {path}")
    viewpoints: list[Viewpoint] = []
    stems: set[str] = set()
    allowed = {
        "CameraPosition",
        "CameraFocalPoint",
        "CameraViewUp",
        "CameraViewAngle",
        "CameraParallelScale",
        "CameraParallelProjection",
        "CenterOfRotation",
        "RotationFactor",
    }
    required = {
        "CameraPosition",
        "CameraFocalPoint",
        "CameraViewUp",
        "CameraParallelScale",
        "CameraParallelProjection",
    }
    for button in root:
        camera = button.find("./Configuration/PVCameraConfiguration/Proxy")
        if camera is None:
            continue
        tooltip = button.find("ToolTip")
        if tooltip is None or not (name := tooltip.get("value", "").strip()):
            raise ValueError("Saved viewpoints must have nonempty names")
        stem = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
        if not stem or stem in stems:
            raise ValueError(f"Viewpoint names must produce unique filenames: {name!r}")
        properties: dict[str, list[float]] = {}
        for prop in camera.findall("Property"):
            key = prop.get("name", "")
            if key not in allowed:
                continue
            values = [float(element.attrib["value"]) for element in prop.findall("Element")]
            count = (
                3
                if key in {"CameraPosition", "CameraFocalPoint", "CameraViewUp", "CenterOfRotation"}
                else 1
            )
            if len(values) != count or not all(map(math.isfinite, values)):
                raise ValueError(f"Invalid {key} in viewpoint {name!r}")
            properties[key] = values
        if required - properties.keys():
            raise ValueError(f"Viewpoint {name!r} is missing camera properties")
        if (
            properties["CameraParallelProjection"][0] not in {0, 1}
            or properties["CameraParallelScale"][0] <= 0
        ):
            raise ValueError(f"Invalid projection settings in viewpoint {name!r}")
        stems.add(stem)
        viewpoints.append(Viewpoint(name, stem, properties))
    if not viewpoints:
        raise ValueError(f"No saved viewpoints in {path}")
    return viewpoints


def data_leaves(data: Any) -> Iterator[Any]:
    """Visit leaf datasets without counting a composite's geometry twice."""
    if data.IsA("vtkCompositeDataSet"):
        iterator = data.NewIterator()
        iterator.SkipEmptyNodesOn()
        if hasattr(iterator, "VisitOnlyLeavesOn"):
            iterator.VisitOnlyLeavesOn()
        iterator.InitTraversal()
        while not iterator.IsDoneWithTraversal():
            yield from data_leaves(iterator.GetCurrentDataObject())
            iterator.GoToNextItem()
    elif hasattr(data, "GetCellData"):
        yield data


def positive_range(source: Any, array: str, *, allow_nan: bool = False) -> tuple[float, int]:
    """Find positive face values, optionally excluding undefined relative errors."""
    minimum = math.inf
    zeros = 0
    for data in data_leaves(servermanager.Fetch(source)):
        values = data.GetCellData().GetArray(array)
        if values is None:
            continue
        if values.GetNumberOfComponents() != 1:
            raise ValueError(f"Export requires a scalar wall field: {array}")
        samples = vtk_to_numpy(values)
        if np.isinf(samples).any() or (not allow_nan and np.isnan(samples).any()):
            raise ValueError(f"{array} contains invalid nonfinite scalar values")
        if np.any(samples < 0):
            raise ValueError(f"{array} must be nonnegative")
        zeros += int(np.count_nonzero(samples == 0))
        positive = samples[samples > 0]
        if positive.size:
            minimum = min(minimum, float(positive.min()))
    if not math.isfinite(minimum):
        raise ValueError(f"Logarithmic export requires positive face values in {array}")
    return minimum, zeros


def template_color_range(path: Path, source_name: str) -> tuple[float, float]:
    """Recover saved scalar limits before automatic data rescaling introduces zero."""
    state = ET.parse(path).getroot().find("ServerManagerState")
    if state is None:
        raise ValueError(f"Not a ParaView state: {path}")
    source = next(
        (
            item
            for item in state.findall("./ProxyCollection[@name='sources']/Item")
            if item.get("name") == source_name
        ),
        None,
    )
    if source is None:
        raise ValueError(f"Template must contain {source_name!r}")
    for proxy in state.findall("Proxy"):
        if proxy.find(f"./Property[@name='Input']/Proxy[@value='{source.get('id')}']") is None:
            continue
        reference = proxy.find("./Property[@name='LookupTable']/Proxy")
        if reference is None:
            continue
        lut = state.find(f"./Proxy[@id='{reference.get('value')}']")
        if lut is None:
            continue
        points = [
            float(e.attrib["value"]) for e in lut.findall("./Property[@name='RGBPoints']/Element")
        ]
        if len(points) >= 8 and len(points) % 4 == 0:
            lower, upper = points[0], points[-4]
            if math.isfinite(lower) and math.isfinite(upper) and lower < upper:
                return lower, upper
    raise ValueError(f"Template has no valid color range for {source_name!r}")


def export_viewpoints(
    view: Any,
    source: Any,
    display: Any,
    directory: Path,
    presets: Path,
    resolution: tuple[int, int],
    *,
    template: Path,
    source_name: str = "Radiation load components",
    filename_prefix: str = "radiation-load",
    allow_nan: bool = False,
    include_log: bool = True,
) -> list[dict[str, Any]]:
    """Keep layout/typography from PVSM and projection from each saved camera."""
    viewpoints = load_viewpoints(presets)
    lut = display.LookupTable
    array = display.ColorArrayName[1]
    if lut.UseLogScale:
        lut.MapControlPointsToLinearSpace()
        lut.UseLogScale = 0
    colors = list(lut.RGBPoints)
    linear_range = (float(colors[0]), float(colors[-4]))
    scales = [("linear", linear_range)]
    zeros: int | None = None
    log_lower_bound_source: str | None = None
    if include_log:
        minimum, zeros = positive_range(source, array, allow_nan=allow_nan)
        configured_minimum, _ = template_color_range(template, source_name)
        log_minimum = configured_minimum if configured_minimum > 0 else minimum
        log_range = (max(log_minimum, linear_range[0]), linear_range[1])
        if log_range[0] == log_range[1]:
            lower = math.nextafter(log_range[0], 0.0)
            log_range = (
                (lower, log_range[1])
                if lower > 0
                else (log_range[0], math.nextafter(log_range[1], math.inf))
            )
        if not 0 < log_range[0] < log_range[1]:
            raise ValueError(
                f"The saved color range for {source_name!r} has no positive logarithmic interval"
            )
        scales.append(("log", log_range))
        log_lower_bound_source = "template" if configured_minimum > 0 else "data"
    lut.AutomaticRescaleRangeMode = "Never"
    low_color = colors[1:4]
    scalar_bar = pv.GetScalarBar(lut, view)
    custom_labels = list(scalar_bar.CustomLabels)
    use_custom_labels = bool(scalar_bar.UseCustomLabels)
    add_range_labels = bool(scalar_bar.AddRangeLabels)
    draw_tick_marks = bool(scalar_bar.DrawTickMarks)
    draw_tick_labels = bool(scalar_bar.DrawTickLabels)
    exports: list[dict[str, Any]] = []
    pv.GetLayout(view).SetSize(*resolution)
    for viewpoint in viewpoints:
        for key, values in viewpoint.properties.items():
            value = values if len(values) > 1 else values[0]
            setattr(view, key, int(value) if key == "CameraParallelProjection" else value)
        view.InteractionMode = 1 if view.CameraParallelProjection else 0
        view.GetClientSideObject().GetRenderer().ResetCameraClippingRange()
        for scale, limits in scales:
            lut.UseLogScale = 0
            lut.RGBPoints = colors
            lut.RescaleTransferFunction(*limits)
            if scale == "log":
                lut.MapControlPointsToLogSpace()
                lut.UseLogScale = 1
                # Native log axes draw decade labels and intervening minor
                # ticks. Custom tick positions replace both, losing subticks.
                scalar_bar.UseCustomLabels = 0
                scalar_bar.AddRangeLabels = 0
                scalar_bar.DrawTickMarks = 1
                scalar_bar.DrawTickLabels = 1
            else:
                scalar_bar.UseCustomLabels = use_custom_labels
                scalar_bar.CustomLabels = custom_labels
                scalar_bar.AddRangeLabels = add_range_labels
                scalar_bar.DrawTickMarks = draw_tick_marks
                scalar_bar.DrawTickLabels = draw_tick_labels
            lut.UseBelowRangeColor = 1
            lut.BelowRangeColor = low_color
            pv.Render(view)
            stem = f"{filename_prefix}-{viewpoint.stem}-{scale}"
            pv.SaveState(str(directory / f"{stem}.pvsm"))
            pv.SaveScreenshot(
                str(directory / f"{stem}.png"),
                view,
                ImageResolution=resolution,
                FontScaling="Scale fonts proportionally",
                TransparentBackground=1,
            )
            if not (directory / f"{stem}.png").is_file():
                raise RuntimeError(f"Screenshot was not written: {stem}.png")
            exports.append(
                dict(
                    viewpoint=viewpoint.name,
                    scale=scale,
                    image=f"{stem}.png",
                    state=f"{stem}.pvsm",
                    resolution=list(resolution),
                    transparent_background=True,
                    color_range=list(limits),
                    log_lower_bound_source=log_lower_bound_source if scale == "log" else None,
                    zero_value_faces=zeros,
                    projection="orthographic" if view.CameraParallelProjection else "perspective",
                    camera=dict(
                        position=list(view.CameraPosition),
                        focal_point=list(view.CameraFocalPoint),
                        view_up=list(view.CameraViewUp),
                        parallel_scale=float(view.CameraParallelScale),
                        parallel_projection=bool(view.CameraParallelProjection),
                    ),
                    scalar_bars=[
                        dict(
                            title=bar.Title,
                            title_font_size=int(bar.TitleFontSize),
                            label_font_size=int(bar.LabelFontSize),
                            thickness=int(bar.ScalarBarThickness),
                            draw_tick_marks=bool(bar.DrawTickMarks),
                            draw_tick_labels=bool(bar.DrawTickLabels),
                            custom_labels=list(bar.CustomLabels) if bar.UseCustomLabels else None,
                        )
                        for bar in view.Representations
                        if getattr(bar, "Visibility", False)
                        and bar.SMProxy.GetXMLName() == "ScalarBarWidgetRepresentation"
                    ],
                )
            )
            print(f"Exported {viewpoint.name} / {scale}: {directory / f'{stem}.png'}", flush=True)
    return exports
