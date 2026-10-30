"""Build the configured emitter and record the actual radiation input times."""

from typing import Any

from cherab.imas.emitter import load_radiation_emitter
from cherab.imas.ids.common import get_ids_time_slice
from imas import DBEntry
from raysect.optical import World

from _configuration import Config, RadiationInputConfig, require_runtime


def _input_identity(query: RadiationInputConfig, time: float, threshold: float) -> dict[str, Any]:
    with DBEntry(query.uri, "r", **query.kwargs) as entry:
        ids = get_ids_time_slice(
            entry, "radiation", time=time, occurrence=query.occurrence, time_threshold=threshold
        )
        if len(ids.time) != 1:
            raise ValueError("Radiation input must resolve to one time slice")
        return dict(
            uri=query.uri,
            occurrence=query.occurrence,
            requested_time=time,
            actual_time=float(ids.time[0]),
        )


def load_source(world: World, config: Config) -> dict[str, Any]:
    """Attach the source to the scene and return serialisable provenance."""
    require_runtime(config)
    radiation = config.radiation
    assert radiation is not None and radiation.primary.time is not None
    primary = radiation.primary
    secondary = radiation.secondary
    primary_info = _input_identity(primary, primary.time, radiation.time_threshold)
    secondary_time = (
        secondary.time if secondary is not None and secondary.time is not None else primary.time
    )
    if secondary is not None and secondary_time == 0 and primary.time != 0:
        # Current upstream uses `time2 or time`, which silently replaces explicit zero.
        raise NotImplementedError(
            "An explicit secondary.time=0 with a nonzero primary.time requires the "
            "cherab-imas time2 fallback fix; refusing to select a different time silently"
        )
    secondary_info = (
        _input_identity(secondary, secondary_time, radiation.time_threshold)
        if secondary is not None
        else None
    )
    emitter = load_radiation_emitter(
        primary.uri,
        time=primary.time,
        occurrence=primary.occurrence,
        args2=(secondary.uri,) if secondary else None,
        kwargs2=secondary.kwargs if secondary else None,
        time2=secondary_time,
        occurrence2=secondary.occurrence if secondary else 0,
        process_index=radiation.process_indices,
        grid_subset_id=radiation.grid_subset_id,
        source=radiation.emissivity_source,
        time_threshold=radiation.time_threshold,
        step=config.emitter.integration_step,
        parent=world,
        interpolator_cache=radiation.interpolator_cache,
        interpolator_cache_dir=radiation.interpolator_cache_dir,
        **primary.kwargs,
    )
    emitter.name = "radiation_emitter"
    return dict(
        kind=config.emitter.kind,
        time=primary_info["actual_time"],
        primary=primary_info,
        secondary=secondary_info,
        process_indices=radiation.process_indices,
        emissivity_source=radiation.emissivity_source,
    )
