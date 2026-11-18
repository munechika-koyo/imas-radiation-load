"""Load a shared PVSM layout with sources from the rendering configuration."""

from __future__ import annotations

import json
import math
import os
import time
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import imas
import imas_paraview
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from paraview import servermanager
from paraview import simple as pv

from _configuration import Config, RadiationInputConfig, hdf5_entry_path
from _paraview_export import export_viewpoints
from _paraview_wall import remove_component_inputs, wall_pipeline
from view_results import CONFIGS_DIRECTORY, load_render_config


def radiation_source(config: Config, *, deferred: bool = False) -> RadiationInputConfig:
    """Find the GGD source when a calculation combines core and edge radiation."""
    assert config.radiation is not None
    candidates = [
        query
        for query in (config.radiation.primary, config.radiation.secondary)
        if query is not None
    ]
    sources = candidates
    if not deferred or len(candidates) > 1:
        sources = []
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


def render_template(path: Path, context: dict[str, str | bool]) -> ET.Element:
    env = Environment(
        loader=FileSystemLoader(path.parent), autoescape=True, undefined=StrictUndefined
    )
    return ET.fromstring(env.get_template(path.name).render(context))


def source_id(state: ET.Element, name: str) -> str:
    item = state.find(f"./ProxyCollection[@name='sources']/Item[@name='{name}']")
    if item is None:
        raise ValueError(f"State must contain a source named {name!r}")
    return item.attrib["id"]


def bind_saved_state(state: ET.Element, context: dict[str, str | bool]) -> None:
    """Rebind a native GUI state, including older states with an empty radiation source."""
    remove_component_inputs(state, source_id(state, "wall IDS"))
    proxy_id = source_id(state, "radiation IDS")
    proxy = state.find(f"./Proxy[@id='{proxy_id}']")
    if proxy is None or proxy.get("type") != "GGDReader":
        default = render_template(CONFIGS_DIRECTORY / "paraview/radiation-load.pvsm.j2", context)
        prototype = default.find("./ServerManagerState/Proxy[@type='GGDReader']")
        assert prototype is not None
        original_id = prototype.attrib["id"]
        for child in prototype.iter():
            if child.get("id", "").partition(".")[0] == original_id:
                child.set("id", proxy_id + child.attrib["id"][len(original_id) :])
        if proxy is not None:
            state.remove(proxy)
        state.append(prototype)
        proxy = prototype
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
        "Enter URI": context["radiation_uri"],
        "IDSAndOccurrence": context["radiation_ids"],
    }.items():
        element = proxy.find(f"./Property[@name='{key}']/Element")
        if element is None:
            raise ValueError(f"Template radiation reader is missing {key!r}")
        element.set("value", str(value))
    # Metadata stays available to the GUI, while geometry and field arrays are
    # requested only when this branch is executed or displayed.
    if not context["show_emissivity"]:
        lazy = proxy.find("./Property[@name='LazyLoading']/Element")
        if lazy is not None:
            lazy.set("value", "1")
        consumers: dict[str, set[str]] = defaultdict(set)
        for node in state.iterfind("Proxy"):
            for input_proxy in node.iterfind("./Property[@name='Input']/Proxy"):
                consumers[input_proxy.attrib["value"]].add(node.attrib["id"])
        deferred = set()
        pending = {proxy_id}
        while pending:
            current = pending.pop()
            deferred.add(current)
            pending.update(consumers[current] - deferred)
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
    if context["export"]:
        for element in state.findall("./Proxy/Property[@name='SuppressLOD']/Element"):
            element.set("value", "1")


def vtk_xml(element: ET.Element) -> Any:
    parser = servermanager.vtkPVXMLParser()
    if not parser.Parse(ET.tostring(element, encoding="unicode")):
        raise ValueError(f"ParaView could not parse {element.tag}")
    return parser.GetRootElement()


def load_scene(template: Path, wall: Any, context: dict[str, str | bool]) -> tuple[Any, ET.Element]:
    """Restore the state with the run's wall source and preserve the live Qt view."""
    native = template.suffix != ".j2"
    root = ET.parse(template).getroot() if native else render_template(template, context)
    state = root.find("ServerManagerState")
    if state is None:
        raise ValueError(f"Not a ParaView state: {template}")
    if native:
        bind_saved_state(state, context)
    loader = servermanager.vtkSMStateLoader()
    loader.SetSession(servermanager.ActiveConnection.Session)
    locator = loader.GetProxyLocator()
    locator.AssignProxy(int(source_id(state, "wall IDS")), wall.SMProxy)
    manager = servermanager.ProxyManager().SMProxyManager
    if context["export"]:
        pv.RemoveViewsAndLayouts()
        manager.LoadXMLState(vtk_xml(root), loader)
        return pv.GetViews()[0], state

    view = pv.GetActiveViewOrCreate("RenderView")
    view_xml = state.find("./Proxy[@type='RenderView']")
    layout_xml = state.find("./Proxy[@type='ViewLayout']")
    if view_xml is None or layout_xml is None:
        raise ValueError("The state template must contain a render view and its layout")
    locator.AssignProxy(int(view_xml.attrib["id"]), view.SMProxy)
    locator.AssignProxy(int(layout_xml.attrib["id"]), pv.GetLayout(view).SMProxy)
    manager.LoadXMLState(vtk_xml(root), loader)
    # Reused views need their properties restored after representations are bound.
    size = view_xml.find("./Property[@name='ViewSize']")
    if size is not None:
        for element, value in zip(size.findall("Element"), view.ViewSize, strict=True):
            element.set("value", str(value))
    if not view.SMProxy.LoadXMLState(vtk_xml(view_xml), locator):
        raise RuntimeError("Could not restore the template render view")
    view.UpdateVTKObjects()
    pv.SetActiveView(view)
    return view, state


WALL_QUANTITIES = {
    "power": ("Radiation load components", "radiation-load"),
    "absolute": ("Absolute error", "radiation-error-absolute"),
    "relative": ("Relative error", "radiation-error-relative"),
}


def select_quantity(view: Any, quantity: str, *, saved_state: bool) -> tuple[Any, Any]:
    """Activate the selected wall field and apply the saved color range policy."""
    if saved_state:
        for key, (name, _) in WALL_QUANTITIES.items():
            branch = pv.FindSource(name)
            if branch is None:
                raise ValueError(f"State must contain a source named {name!r}")
            display = pv.GetDisplayProperties(branch, view)
            display.Visibility = int(key == quantity)
            display.SetScalarBarVisibility(view, key == quantity)
    name, _ = WALL_QUANTITIES[quantity]
    source = pv.FindSource(name)
    if source is None:
        raise ValueError(f"State must contain a source named {name!r}")
    source.UpdatePipeline()
    display = pv.GetDisplayProperties(source, view)
    array = display.ColorArrayName[1]
    if array not in source.CellData.keys():
        raise ValueError(f"Template wall field {array!r} is unavailable")
    if display.LookupTable.AutomaticRescaleRangeMode != "Never":
        display.RescaleTransferFunctionToDataRange(False, True)
    pv.SetActiveView(view)
    pv.SetActiveSource(source)
    return source, display


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
    mode = os.environ["RADLOAD_VIEW_MODE"]
    view, state = load_scene(
        template,
        wall,
        dict(
            radiation_uri=radiation.uri,
            radiation_ids=f"radiation/{radiation.occurrence}"
            if radiation.occurrence
            else "radiation",
            show_emissivity=show_emissivity,
            quantity=quantity,
            export=mode == "export",
        ),
    )
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
        emission_times = emission.GetProperty("TimestepValues")[:]
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
        if template.suffix != ".j2":
            emission_display.Visibility = 1
            emission_display.SetScalarBarVisibility(view, True)
        emission_report.update(
            array=emission_array,
            cells=plane.GetDataInformation().GetNumberOfCells(),
            opacity=emission_display.Opacity,
            log_scale=bool(emission_display.LookupTable.UseLogScale),
            loaded=True,
        )
    export_size = tuple(view.ViewSize)
    if mode == "export":
        pv.GetLayout(view).SetSize(*export_size)
    view.ViewTime = selected_time
    animation = pv.GetAnimationScene()
    animation.UpdateAnimationUsingDataTimeSteps()
    animation.AnimationTime = selected_time
    caption = pv.FindSource("Time slice")
    if caption is not None:
        caption.Text = f"t = {selected_time:.7g} s"
    source, display = select_quantity(view, quantity, saved_state=template.suffix != ".j2")
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
                state=state,
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
