"""Load a shared PVSM layout with sources from the rendering configuration."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any

import imas
import imas_paraview
from paraview import servermanager
from paraview import simple as pv

from _configuration import Config, RadiationInputConfig, hdf5_entry_path
from _paraview_export import export_viewpoints
from _paraview_wall import remove_component_inputs, wall_pipeline
from view_results import CONFIGS_DIRECTORY, TEMP_DIRECTORY, load_render_config


def radiation_source(config: Config, *, deferred: bool = False) -> RadiationInputConfig:
    """Find the GGD source when a calculation combines core and edge radiation."""
    assert config.radiation is not None
    candidates = [
        query
        for query in (config.radiation.primary, config.radiation.secondary)
        if query is not None
    ]
    if deferred and len(candidates) == 1:
        source = candidates[0]
        if source.kwargs:
            raise ValueError(
                "The ParaView GGD reader cannot apply radiation DBEntry kwargs; use a self-contained URI"
            )
        return source
    sources: list[RadiationInputConfig] = []
    for query in candidates:
        with imas.DBEntry(query.uri, "r", **query.kwargs) as entry:
            radiation = entry.get("radiation", occurrence=query.occurrence, lazy=True)
            if len(radiation.grid_ggd):
                sources.append(query)
    if len(sources) != 1:
        raise ValueError(
            f"The shared scene requires exactly one GGD radiation source; found {len(sources)}"
        )
    source = sources[0]
    if source.kwargs:
        raise ValueError(
            "The ParaView GGD reader cannot apply radiation DBEntry kwargs; use a self-contained URI"
        )
    return source


def load_gui_state(path: Path, state: ET.Element, bindings: dict[int, Any] | None = None) -> Any:
    """Restore a state into the existing Qt view without replacing its render widget."""
    view = pv.GetActiveViewOrCreate("RenderView")
    layout = pv.GetLayout(view)
    view_xml = state.find("./Proxy[@type='RenderView']")
    layout_xml = state.find("./Proxy[@type='ViewLayout']")
    if view_xml is None or layout_xml is None:
        raise ValueError("The state template must contain a render view and its layout")
    loader = servermanager.vtkSMStateLoader()
    loader.SetSession(servermanager.ActiveConnection.Session)
    for proxy_id, proxy in (bindings or {}).items():
        loader.GetProxyLocator().AssignProxy(proxy_id, proxy.SMProxy)
    for element, proxy in ((view_xml, view), (layout_xml, layout)):
        loader.GetProxyLocator().AssignProxy(int(element.attrib["id"]), proxy.SMProxy)
    servermanager.ProxyManager().LoadState(str(path), loader)
    # Assigned proxies are reused without restoring their XML properties. Apply
    # the view settings explicitly once representation references can be resolved.
    # Keep Qt's actual viewport dimensions rather than the saved export resolution.
    size = view_xml.find("./Property[@name='ViewSize']")
    if size is not None:
        for element, value in zip(size.findall("Element"), view.ViewSize, strict=True):
            element.set("value", str(value))
    parser = servermanager.vtkPVXMLParser()
    if not parser.Parse(ET.tostring(view_xml, encoding="unicode")):
        raise ValueError("Could not parse the template render view")
    if not view.SMProxy.LoadXMLState(parser.GetRootElement(), loader.GetProxyLocator()):
        raise RuntimeError("Could not restore the template render view")
    view.UpdateVTKObjects()
    pv.SetActiveView(view)
    return view


def load_state_tree(tree: ET.ElementTree, bindings: dict[int, Any] | None = None) -> Any:
    """Load a bound state, preserving the live Qt view in interactive sessions."""
    state = tree.getroot().find("ServerManagerState")
    if state is None:
        raise ValueError("Not a ParaView state")
    if os.environ["RADLOAD_VIEW_MODE"] == "export":
        # Export full geometry. VTK's quadric LOD filter can also crash when
        # reducing these composite grids, even with a high view LOD threshold.
        for element in state.findall("./Proxy/Property[@name='SuppressLOD']/Element"):
            element.set("value", "1")
    TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".radiation-view-state-", dir=TEMP_DIRECTORY
    ) as directory:
        path = Path(directory) / "scene.pvsm"
        tree.write(path, encoding="utf-8", xml_declaration=True)
        if os.environ["RADLOAD_VIEW_MODE"] == "gui":
            return load_gui_state(path, state, bindings)
        if bindings:
            pv.RemoveViewsAndLayouts()
            loader = servermanager.vtkSMStateLoader()
            loader.SetSession(servermanager.ActiveConnection.Session)
            for proxy_id, proxy in bindings.items():
                loader.GetProxyLocator().AssignProxy(proxy_id, proxy.SMProxy)
            servermanager.ProxyManager().LoadState(str(path), loader)
        else:
            pv.LoadState(str(path))
    return pv.GetViews()[0]


def load_template(
    template: Path, radiation: RadiationInputConfig, wall: Any, *, show_emissivity: bool
) -> Any:
    """Bind readers and initial visibility before any representation requests data."""
    tree = ET.parse(template)
    state = tree.getroot().find("ServerManagerState")
    if state is None:
        raise ValueError(f"Not a ParaView state: {template}")
    wall_item = state.find("./ProxyCollection[@name='sources']/Item[@name='wall IDS']")
    if wall_item is None:
        raise ValueError("Template must contain a source named 'wall IDS'")
    wall_id = int(wall_item.attrib["id"])
    remove_component_inputs(state, wall_id)
    bindings = {wall_id: wall}
    item = state.find("./ProxyCollection[@name='sources']/Item[@name='radiation IDS']")
    if item is None:
        raise ValueError("Template must contain a reader named 'radiation IDS'")
    proxy_id = int(item.attrib["id"])
    proxy = state.find(f"./Proxy[@id='{proxy_id}']")
    if proxy is None or proxy.get("type") != "GGDReader":
        # A wall-only saved state has an empty source at this binding point.
        default = ET.parse(CONFIGS_DIRECTORY / "paraview/radiation-load.pvsm")
        default_state = default.getroot().find("ServerManagerState")
        assert default_state is not None
        reader = default_state.find(
            "./ProxyCollection[@name='sources']/Item[@name='radiation IDS']"
        )
        assert reader is not None
        original_id = reader.attrib["id"]
        prototype = default_state.find(f"./Proxy[@id='{original_id}']")
        assert prototype is not None
        if proxy is not None:
            state.remove(proxy)
        proxy = deepcopy(prototype)
        for child in proxy.iter():
            if child.get("id", "").split(".", 1)[0] == original_id:
                child.set("id", str(proxy_id) + child.attrib["id"][len(original_id) :])
        state.append(proxy)
    selector = proxy.find("./Property[@name='AttributeArraySelector']")
    if selector is not None:
        elements = selector.findall("Element")
        for key, enabled in zip(elements[::2], elements[1::2], strict=True):
            if enabled.get("value") not in {"0", "1"}:
                # ParaView 6.1 can save name/name pairs for this plugin.
                if enabled.get("value") != key.get("value"):
                    raise ValueError("Invalid saved radiation array selection")
                enabled.set("value", "1")
    for key, value in {
        "Enter URI": radiation.uri,
        "IDSAndOccurrence": "radiation"
        if radiation.occurrence == 0
        else f"radiation/{radiation.occurrence}",
    }.items():
        element = proxy.find(f"./Property[@name='{key}']/Element")
        if element is None:
            raise ValueError(f"Template radiation reader is missing {key!r}")
        element.set("value", value)
    # Metadata stays available to the GUI, while geometry and field arrays are
    # requested only when this branch is executed or displayed.
    if not show_emissivity:
        lazy = proxy.find("./Property[@name='LazyLoading']/Element")
        if lazy is not None:
            lazy.set("value", "1")
        deferred = {str(proxy_id)}
        for _ in range(len(state.findall("Proxy"))):
            descendants = {
                node.attrib["id"]
                for node in state.findall("Proxy")
                if any(
                    p.get("value") in deferred
                    for p in node.findall("./Property[@name='Input']/Proxy")
                )
            }
            if descendants <= deferred:
                break
            deferred.update(descendants)
        tables: set[str] = set()
        for node in state.findall("Proxy"):
            if node.get("id") in deferred and node.get("group") == "representations":
                visibility = node.find("./Property[@name='Visibility']/Element")
                if visibility is not None:
                    visibility.set("value", "0")
                table = node.find("./Property[@name='LookupTable']/Proxy")
                if table is not None:
                    tables.add(table.attrib["value"])
        for node in state.findall("./Proxy[@type='ScalarBarWidgetRepresentation']"):
            table = node.find("./Property[@name='LookupTable']/Proxy")
            if table is not None and table.get("value") in tables:
                node.find("./Property[@name='Visibility']/Element").set("value", "0")
    return load_state_tree(tree, bindings)


WALL_QUANTITIES = {
    "power": ("Radiation load components", "radiation-load"),
    "absolute": ("Absolute error", "radiation-error-absolute"),
    "relative": ("Relative error", "radiation-error-relative"),
}


def select_quantity(view: Any, quantity: str) -> tuple[Any, Any]:
    """Show one wall branch and its scalar bar while keeping the camera and overlay."""
    if quantity not in WALL_QUANTITIES:
        raise ValueError("Use the power/absolute/relative quantities defined in view-results")
    selected: tuple[Any, Any] | None = None
    for name, (source_name, _) in WALL_QUANTITIES.items():
        source = pv.FindSource(source_name)
        if source is None:
            raise ValueError(f"Template must contain a source named {source_name!r}")
        source.UpdatePipeline()
        display = pv.GetDisplayProperties(source, view)
        visible = name == quantity
        display.Visibility = int(visible)
        display.SetScalarBarVisibility(view, visible)
        if visible:
            array = display.ColorArrayName[1]
            if array not in source.CellData.keys():
                raise ValueError(f"Template wall field {array!r} is unavailable")
            if display.LookupTable.AutomaticRescaleRangeMode != "Never":
                display.RescaleTransferFunctionToDataRange(False, True)
            selected = source, display
    assert selected is not None
    pv.SetActiveView(view)
    pv.SetActiveSource(selected[0])
    return selected


def times(source: Any) -> list[float]:
    values = source.GetProperty("TimestepValues")
    return [float(values[i]) for i in range(len(values))]


def build() -> dict[str, Any]:
    start = time.monotonic()
    config = load_render_config(Path(os.environ["RADLOAD_VIEW_CONFIG"]))
    template = Path(os.environ["RADLOAD_VIEW_TEMPLATE"])
    quantity = os.environ["RADLOAD_VIEW_QUANTITY"]
    if quantity not in WALL_QUANTITIES:
        raise ValueError("Use the power/absolute/relative quantities defined in view-results")
    emissivity = os.environ["RADLOAD_VIEW_EMISSIVITY"]
    if emissivity not in {"true", "false"}:
        raise ValueError("Use the true/false emissivity options defined in view-results")
    show_emissivity = emissivity == "true"
    radiation = radiation_source(config, deferred=not show_emissivity)
    pv._DisableFirstRenderCameraReset()
    presets = CONFIGS_DIRECTORY / "colormaps/ultraplot-fire.json"
    if not pv.ImportPresets(str(presets)):
        raise RuntimeError(f"Could not import color presets: {presets}")
    wall, statistics, selected_time = wall_pipeline(config.output.uri, config.output.occurrence)
    manifest_path = hdf5_entry_path(config.output.uri) / "rendering.json"
    sampling = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if sampling is not None:
        expected = set(sampling["components"])
        actual = set(statistics)
        if expected != actual:
            raise ValueError(
                "Measured wall IDS subsets differ from rendering.json components: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )
    if wall.GetDataInformation().GetNumberOfCells() != sum(
        item["cells"] for item in statistics.values()
    ):
        raise ValueError("The wall scene does not contain every measured component face")
    pv.LoadPlugin(str(Path(imas_paraview.__path__[0]) / "plugins/vtkggdreader.py"), remote=False)
    view = load_template(template, radiation, wall, show_emissivity=show_emissivity)
    plane = pv.FindSource("Emissivity cells")
    emission_report: dict[str, Any] = dict(
        uri=radiation.uri,
        occurrence=radiation.occurrence,
        loaded=False,
        initially_visible=show_emissivity,
    )
    if show_emissivity:
        emission = pv.FindSource("radiation IDS")
        if emission is None or plane is None:
            raise ValueError("Template must contain 'radiation IDS' and 'Emissivity cells'")
        emission.UpdatePipelineInformation()
        emission_times = times(emission)
        if not emission_times:
            raise ValueError("Radiation IDS must provide a time slice")
        emission_time = min(emission_times, key=lambda t: abs(t - selected_time))
        if not math.isclose(emission_time, selected_time, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"No common wall/radiation time slice: wall={selected_time}, radiation={emission_times}"
            )
        plane.UpdatePipeline(selected_time)
        if plane.GetDataInformation().GetNumberOfCells() == 0:
            raise ValueError("No radiation cells to display with this template")
        emission_display = pv.GetDisplayProperties(plane, view)
        emission_array = emission_display.ColorArrayName[1]
        if emission_array not in plane.CellData.keys():
            raise ValueError(
                f"Template emissivity field {emission_array!r} is unavailable; change it in the state template"
            )
        emission_display.Visibility = int(emissivity == "true")
        emission_display.SetScalarBarVisibility(view, emissivity == "true")
        emission_report = dict(
            uri=radiation.uri,
            occurrence=radiation.occurrence,
            array=emission_array,
            cells=plane.GetDataInformation().GetNumberOfCells(),
            opacity=emission_display.Opacity,
            log_scale=bool(emission_display.LookupTable.UseLogScale),
            initially_visible=True,
            loaded=True,
        )
    elif plane is not None:
        display = pv.GetDisplayProperties(plane, view)
        display.Visibility = 0
        display.SetScalarBarVisibility(view, False)
    export_size = tuple(view.ViewSize)
    mode = os.environ["RADLOAD_VIEW_MODE"]
    if mode == "export":
        pv.GetLayout(view).SetSize(*export_size)
    view.ViewTime = selected_time
    animation = pv.GetAnimationScene()
    animation.UpdateAnimationUsingDataTimeSteps()
    animation.AnimationTime = selected_time
    caption = pv.FindSource("Time slice")
    if caption is not None:
        caption.Text = f"t = {selected_time:.7g} s"
    source, display = select_quantity(view, quantity)
    pv.Render(view)
    colors = list(display.LookupTable.RGBPoints)
    report = dict(
        status="ready",
        config=str(config.source),
        template=str(template),
        quantity=quantity,
        time=selected_time,
        view=dict(
            mode=mode,
            size=list(view.ViewSize),
            render_window_size=list(view.GetClientSideObject().GetRenderWindow().GetSize()),
            uses_color_palette=bool(view.UseColorPaletteForBackground),
            background=list(view.GetClientSideObject().GetRenderer().GetBackground()),
        ),
        wall=dict(
            uri=config.output.uri,
            occurrence=config.output.occurrence,
            array=display.ColorArrayName[1],
            cells=source.GetDataInformation().GetNumberOfCells(),
            color_range=[float(colors[0]), float(colors[-4])],
            components=statistics,
        ),
        emissivity=emission_report,
        sampling=sampling,
        uncertainty="1-sigma Monte Carlo standard error of the mean [W m-2]",
        relative_error="100 * standard_error / abs(power_density); NaN at zero load",
        elapsed_seconds=time.monotonic() - start,
    )
    if destination := os.environ.get("RADLOAD_VIEW_OUTPUT"):
        directory = Path(destination)
        if mode == "export":
            source_name, prefix = WALL_QUANTITIES[quantity]
            report["exports"] = export_viewpoints(
                view,
                source,
                display,
                directory,
                Path(os.environ["RADLOAD_VIEW_VIEWPOINTS"]),
                export_size,
                template=template,
                source_name=source_name,
                filename_prefix=prefix,
                allow_nan=quantity == "relative",
                include_log=quantity == "power",
            )
            report["viewpoints"] = os.environ["RADLOAD_VIEW_VIEWPOINTS"]
        else:
            pv.SaveState(str(directory / "scene.pvsm"))
            pv.SaveScreenshot(str(directory / "scene.png"), view)
        report["elapsed_seconds"] = time.monotonic() - start
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if mode == "gui":
            pv.Render(view)
    print(
        f"Ready: {quantity} at t={selected_time:.7g} s ({len(statistics)} components)", flush=True
    )
    return report


if __name__ == "__main__":
    # ProgrammableSource populates __main__ with NumPy names. Keep scene
    # functions in a module namespace so Python builtins retain their meaning.
    from _paraview_scene import build as build_scene

    try:
        build_scene()
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
