"""Load a shared PVSM layout with sources from the rendering configuration."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import imas
import imas_paraview
from paraview import servermanager
from paraview import simple as pv

from _configuration import Config, RadiationInputConfig
from view_results import CONFIGS_DIRECTORY, load_render_config


def radiation_source(config: Config) -> RadiationInputConfig:
    """Find the GGD source when a calculation combines core and edge radiation."""
    assert config.radiation is not None
    candidates = (config.radiation.primary, config.radiation.secondary)
    sources: list[RadiationInputConfig] = []
    for query in candidates:
        if query is None:
            continue
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
    with tempfile.TemporaryDirectory(prefix="radiation-view-state-") as directory:
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


def load_template(template: Path, config: Config, radiation: RadiationInputConfig) -> Any:
    """Bind URI/occurrence properties before ParaView opens the data entries.

    Reader names are the template's binding points. GUI-saved states with
    concrete URIs also work, provided these reader names remain unchanged.
    """
    tree = ET.parse(template)
    state = tree.getroot().find("ServerManagerState")
    if state is None:
        raise ValueError(f"Not a ParaView state: {template}")
    bindings = {
        "wall IDS": (config.output.uri, "wall", config.output.occurrence),
        "radiation IDS": (radiation.uri, "radiation", radiation.occurrence),
    }
    for name, (uri, ids, occurrence) in bindings.items():
        item = state.find(f"./ProxyCollection[@name='sources']/Item[@name='{name}']")
        if item is None:
            raise ValueError(f"Template must contain a reader named {name!r}")
        proxy = state.find(f"./Proxy[@id='{item.get('id')}']")
        if proxy is None or proxy.get("type") != "GGDReader":
            raise ValueError(f"Template source {name!r} must be a GGDReader")
        selector = proxy.find("./Property[@name='AttributeArraySelector']")
        if selector is not None:
            elements = selector.findall("Element")
            for key, enabled in zip(elements[::2], elements[1::2], strict=True):
                if enabled.get("value") not in {"0", "1"}:
                    # ParaView 6.1 can save this plugin's array selection as
                    # name/name pairs. Restore enabled flags so the invalid
                    # integer cannot abort the entire reader property update.
                    if enabled.get("value") != key.get("value"):
                        raise ValueError(f"Invalid saved array selection in {name!r}")
                    enabled.set("value", "1")
        values = {
            "Enter URI": uri,
            "IDSAndOccurrence": ids if occurrence == 0 else f"{ids}/{occurrence}",
        }
        for key, value in values.items():
            element = proxy.find(f"./Property[@name='{key}']/Element")
            if element is None:
                raise ValueError(f"Template reader {name!r} is missing {key!r}")
            element.set("value", value)
    # URI strings are custom plugin properties, so LoadState's filename relocation
    # does not apply. Write the bound state temporarily, leaving the template intact.
    return load_state_tree(tree)


def times(source: Any) -> list[float]:
    values = source.GetProperty("TimestepValues")
    return [float(values[i]) for i in range(len(values))]


def build() -> dict[str, Any]:
    start = time.monotonic()
    config = load_render_config(Path(os.environ["RADLOAD_VIEW_CONFIG"]))
    template = Path(os.environ["RADLOAD_VIEW_TEMPLATE"])
    radiation = radiation_source(config)
    pv._DisableFirstRenderCameraReset()
    presets = CONFIGS_DIRECTORY / "colormaps/ultraplot-fire.json"
    if not pv.ImportPresets(str(presets)):
        raise RuntimeError(f"Could not import color presets: {presets}")
    pv.LoadPlugin(str(Path(imas_paraview.__path__[0]) / "plugins/vtkggdreader.py"), remote=False)
    view = load_template(template, config, radiation)
    wall = pv.FindSource("wall IDS")
    emission = pv.FindSource("radiation IDS")
    components = pv.FindSource("Radiation load components")
    plane = pv.FindSource("Emissivity cells")
    if any(source is None for source in (wall, emission, components, plane)):
        raise ValueError(
            "Template is missing the wall/radiation readers or their component/cell filters"
        )
    wall.UpdatePipelineInformation()
    emission.UpdatePipelineInformation()
    wall_times, emission_times = times(wall), times(emission)
    if not wall_times or not emission_times:
        raise ValueError("Both IDSs must provide a time slice")
    assert config.radiation is not None and config.radiation.primary.time is not None
    selected = min(wall_times, key=lambda t: abs(t - config.radiation.primary.time))
    emission_time = min(emission_times, key=lambda t: abs(t - selected))
    if not math.isclose(emission_time, selected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"No common wall/radiation time slice: wall={selected}, radiation={emission_times}"
        )
    wall.UpdatePipeline(selected)
    assembly = wall.GetDataInformation().GetDataAssembly()
    if assembly is None:
        raise ValueError("Wall IDS has no component grid_subset assembly")
    children = [assembly.GetChild(0, i) for i in range(assembly.GetNumberOfChildren(0))]
    # Exclude the whole-mesh partition, which duplicates the component subsets.
    subsets = [node for node in children if 0 not in assembly.GetDataSetIndices(node, False)]
    components.Selectors = [
        f"/{assembly.GetNodeName(0)}/{assembly.GetNodeName(node)}" for node in subsets or children
    ]
    components.UpdatePipeline(selected)
    plane.UpdatePipeline(selected)
    for name, source in (("wall", components), ("radiation", plane)):
        if source.GetDataInformation().GetNumberOfCells() == 0:
            raise ValueError(f"No {name} cells to display with this template")
    export_size = tuple(view.ViewSize)
    mode = os.environ["RADLOAD_VIEW_MODE"]
    # GUI dimensions belong to Qt; only impose the template's resolution offscreen.
    if mode == "export":
        pv.GetLayout(view).SetSize(*export_size)
    view.ViewTime = selected
    animation = pv.GetAnimationScene()
    animation.UpdateAnimationUsingDataTimeSteps()
    animation.AnimationTime = selected
    caption = pv.FindSource("Time slice")
    if caption is not None:
        caption.Text = f"t = {selected:.7g} s"
    # Keep colors/styles from the template; adapt the wall range to the new result.
    wall_display = pv.GetDisplayProperties(components, view)
    array = wall_display.ColorArrayName[1]
    if array not in components.CellData.keys():
        raise ValueError(f"Template wall field {array!r} is unavailable")
    limits = tuple(components.CellData[array].GetRange(0))
    if not all(math.isfinite(v) for v in limits):
        raise ValueError(f"No finite wall data in {array!r}")
    if wall_display.LookupTable.AutomaticRescaleRangeMode != "Never":
        wall_display.RescaleTransferFunctionToDataRange(False, True)
    emission_display = pv.GetDisplayProperties(plane, view)
    emission_array = emission_display.ColorArrayName[1]
    if emission_array not in plane.CellData.keys():
        raise ValueError(
            f"Template emissivity field {emission_array!r} is unavailable; change it in the state template"
        )
    pv.SetActiveView(view)
    pv.SetActiveSource(components)
    pv.Render(view)
    report = dict(
        status="ready",
        config=str(config.source),
        template=str(template),
        time=selected,
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
            array=array,
            cells=components.GetDataInformation().GetNumberOfCells(),
            color_range=limits,
        ),
        emissivity=dict(
            uri=radiation.uri,
            occurrence=radiation.occurrence,
            array=emission_array,
            cells=plane.GetDataInformation().GetNumberOfCells(),
            opacity=emission_display.Opacity,
            log_scale=bool(emission_display.LookupTable.UseLogScale),
        ),
        elapsed_seconds=time.monotonic() - start,
    )
    if destination := os.environ.get("RADLOAD_VIEW_OUTPUT"):
        directory = Path(destination)
        pv.SaveState(str(directory / "scene.pvsm"))
        options = dict(ImageResolution=export_size) if mode == "export" else {}
        pv.SaveScreenshot(str(directory / "scene.png"), view, **options)
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if mode == "gui":
            pv.Render(view)
    print(f"Ready: radiation load + emissivity at t={selected:.7g} s", flush=True)
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
