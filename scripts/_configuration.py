"""Load, validate and snapshot the radiation-load workflow's TOML configuration."""

from __future__ import annotations

import json
import math
import re
import tomllib
import types
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin, get_type_hints
from urllib.parse import urlsplit

T = TypeVar("T")


@dataclass(frozen=True, kw_only=True)
class EmitterConfig:
    """Select one emission model for a run and its volume integration step."""

    kind: Literal["radiation_ids", "plasma_spectral", "plasma_total"] = "radiation_ids"
    integration_step: float = 1.0e-3


@dataclass(frozen=True, kw_only=True)
class RadiationInputConfig:
    """One radiation IDS query; secondary time may inherit primary time."""

    uri: str
    time: float | None = None
    occurrence: int = 0
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class RadiationConfig:
    """One or two complementary core/GGD emissivity sources, blended by the API."""

    primary: RadiationInputConfig
    secondary: RadiationInputConfig | None = None
    emissivity_source: Literal["auto", "values", "coefficients"] = "values"
    time_threshold: float = math.inf
    process_indices: tuple[int, ...] | None = None
    grid_subset_id: int = 5
    interpolator_cache: Literal["none", "memory", "disk"] = "disk"
    interpolator_cache_dir: Path | None = None


@dataclass(frozen=True, kw_only=True)
class PlasmaConfig:
    """IMAS profile sources, times and core/edge blending."""

    core_uri: str
    time: float
    time_threshold: float
    edge_uri: str | None = None
    time_edge: float | None = None
    blend_psi: tuple[float, float] | None = None
    core_kwargs: dict[str, Any] = field(default_factory=dict)
    edge_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class AtomicConfig:
    """Selected line-emitting elements and the local OpenADAS repository."""

    line_elements: tuple[str, ...] = ()
    total_elements: tuple[str, ...] = ()
    repository_path: Path | None = None


@dataclass(frozen=True, kw_only=True)
class GeometryObjectConfig:
    """One wall IDS geometry source and its toroidal repetition."""

    uri: str
    description_index: int = 0
    occurrence: int = 0
    time: float = 0.0
    num_segments: int = 1


@dataclass(frozen=True, kw_only=True)
class NonTargetConfig(GeometryObjectConfig):
    """Named geometry used only to absorb and occlude rays."""

    name: str


@dataclass(frozen=True, kw_only=True)
class GeometryConfig:
    """Measured wall parts and optional non-target geometry sources."""

    target: GeometryObjectConfig
    non_targets: tuple[NonTargetConfig, ...] = ()


@dataclass(frozen=True, kw_only=True)
class RenderConfig:
    """Spectral resolution, adaptive sampling and result-save frequency."""

    wavelength_bands: tuple[tuple[float, float], ...] = ()
    spectral_bins: int | None = None
    pixel_samples: int
    min_samples: int
    max_passes: int
    adaptive_fraction: float
    adaptive_ratio: float
    adaptive_cutoff: float
    spectral_percentile: float | None = None
    save_every: int
    surface_offset: float = 1.0e-6


@dataclass(frozen=True, kw_only=True)
class OutputConfig:
    """Final IMAS entry and optional persistent storage for intermediate results."""

    uri: str
    occurrence: int = 0
    work_directory: Path | None = None


@dataclass(frozen=True, kw_only=True)
class MergeJobConfig:
    """Resource overrides for the single-writer assembly job."""

    cpus_per_task: int = 1
    memory: str | None = None
    time: str | None = None


@dataclass(frozen=True, kw_only=True)
class SlurmConfig:
    """Template, common allocations and optional assembly-job overrides."""

    job_template: Path
    working_directory: Path
    cpus_per_task: int
    partition: str
    time: str
    memory: str | None = None
    account: str | None = None
    qos: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    merge: MergeJobConfig = field(default_factory=MergeJobConfig)


@dataclass(frozen=True, kw_only=True)
class Config:
    """Typed configuration with absolute paths and its original source path."""

    source: Path
    plasma: PlasmaConfig | None = None
    atomic: AtomicConfig | None = None
    geometry: GeometryConfig
    render: RenderConfig
    output: OutputConfig
    slurm: SlurmConfig
    emitter: EmitterConfig = field(default_factory=EmitterConfig)
    radiation: RadiationConfig | None = None


def _decode(value: Any, annotation: Any, name: str, base: Path) -> Any:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is types.UnionType:
        return _decode(value, next(t for t in arguments if t is not type(None)), name, base)
    if origin is Literal:
        if not any(type(value) is type(choice) and value == choice for choice in arguments):
            raise ValueError(f"{name} must be one of {arguments}")
        return value
    if is_dataclass(annotation):
        return _section(annotation, value, name, base)
    if annotation is Path:
        value = _decode(value, str, name, base)
        path = Path(value).expanduser()
        return (base / path).resolve() if not path.is_absolute() else path.resolve()
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError(f"{name} must be an array")
        if arguments[-1] is Ellipsis:
            return tuple(
                _decode(v, arguments[0], f"{name}[{i}]", base) for i, v in enumerate(value)
            )
        if len(value) != len(arguments):
            raise ValueError(f"{name} must have {len(arguments)} values")
        return tuple(
            _decode(v, t, f"{name}[{i}]", base)
            for i, (v, t) in enumerate(zip(value, arguments, strict=True))
        )
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a table")
        return value.copy()
    if annotation is float and type(value) in (int, float):
        return float(value)
    if type(value) is not annotation:
        raise ValueError(f"{name} must be {annotation.__name__}, got {type(value).__name__}")
    if annotation is str and not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _section(cls: type[T], values: dict[str, Any], name: str, base: Path) -> T:
    if not isinstance(values, dict):
        raise ValueError(f"[{name}] must be a table")
    hints = get_type_hints(cls)
    unknown = values.keys() - hints.keys()
    if unknown:
        raise ValueError(f"Unknown setting in [{name}]: {', '.join(sorted(unknown))}")
    for item in fields(cls):
        if item.name not in values and item.default is MISSING and item.default_factory is MISSING:
            raise ValueError(f"Missing setting: {name}.{item.name}")
    return cls(
        **{
            key: _decode(value, hints[key], f"{name}.{key}" if name else key, base)
            for key, value in values.items()
        }
    )


def _uri(value: str, base: Path) -> str:
    if value.startswith("imas:"):
        return re.sub(
            r"([?;&]path=)([^;&]+)",
            lambda match: match[1] + str(_decode(match[2], Path, "IMAS path", base)),
            value,
        )
    return str(_decode(value, Path, "IMAS URI", base))


def hdf5_entry_path(uri: str) -> Path:
    """Return the directory of a local HDF5 IMAS URI without opening the entry."""
    parsed = urlsplit(uri)
    paths = [
        option.partition("=")[2]
        for option in re.split(r"[;&?]", parsed.query)
        if option.partition("=")[0] == "path"
    ]
    if (
        parsed.scheme != "imas"
        or parsed.path != "hdf5"
        or parsed.netloc
        or parsed.fragment
        or len(paths) != 1
        or not paths[0].strip()
    ):
        raise ValueError("output.uri must be a local HDF5 IMAS URI: imas:hdf5?path=...")
    return Path(paths[0]).expanduser().resolve()


def resolve_output_uri(value: str, base: Path) -> str:
    """Resolve the output entry path relative to the configuration or task directory."""
    hdf5_entry_path(value)
    return _uri(value, base)


def _validate_source(config: Config) -> None:
    """Validate required source tables and parameters supported by the IMAS API."""
    emitter = config.emitter
    if emitter.kind not in {"radiation_ids", "plasma_spectral", "plasma_total"}:
        raise ValueError("Unknown emitter.kind")
    if not math.isfinite(emitter.integration_step) or emitter.integration_step <= 0:
        raise ValueError("emitter.integration_step must be finite and positive")
    if emitter.kind != "radiation_ids":
        if config.radiation is not None:
            raise ValueError("[radiation] is only allowed for emitter.kind = radiation_ids")
        if config.plasma is None or config.atomic is None:
            raise ValueError("Plasma emitters require [plasma] and [atomic]")
        if emitter.kind == "plasma_spectral" and config.atomic.total_elements:
            raise ValueError("atomic.total_elements is only allowed for plasma_total")
        if emitter.kind == "plasma_total":
            if not config.atomic.total_elements:
                raise ValueError("plasma_total requires nonempty atomic.total_elements")
            if config.atomic.line_elements:
                raise ValueError("atomic.line_elements is only allowed for plasma_spectral")
        return
    if config.plasma is not None or config.atomic is not None:
        raise ValueError("radiation_ids does not use [plasma] or [atomic]; omit those tables")
    radiation = config.radiation
    if radiation is None:
        raise ValueError("radiation_ids requires [radiation.primary]")
    if radiation.primary.time is None:
        raise ValueError("radiation.primary.time is required")
    if radiation.secondary is not None and radiation.emissivity_source != "values":
        raise ValueError("Two radiation sources require radiation.emissivity_source = values")
    if math.isnan(radiation.time_threshold) or radiation.time_threshold < 0:
        raise ValueError("radiation.time_threshold must be nonnegative (inf is allowed)")
    if radiation.process_indices is not None and (
        not radiation.process_indices
        or len(set(radiation.process_indices)) != len(radiation.process_indices)
    ):
        raise ValueError("radiation.process_indices must be nonempty and unique when specified")
    reserved = {
        "uri",
        "mode",
        "args",
        "args2",
        "kwargs2",
        "time",
        "time2",
        "occurrence",
        "occurrence2",
        "process_index",
        "grid_ggd",
        "grid_subset_id",
        "equilibrium",
        "psi_interpolator",
        "mask",
        "num_toroidal",
        "source",
        "time_threshold",
        "step",
        "parent",
        "interpolator_cache",
        "interpolator_cache_dir",
    }
    for name, query in (("primary", radiation.primary), ("secondary", radiation.secondary)):
        if query is None:
            continue
        if not query.uri.strip():
            raise ValueError(f"radiation.{name}.uri must not be empty")
        if query.time is not None and (not math.isfinite(query.time) or query.time < 0):
            raise ValueError(f"radiation.{name}.time must be finite and nonnegative")
        if query.occurrence < 0:
            raise ValueError(f"radiation.{name}.occurrence must be nonnegative")
        if duplicates := reserved.intersection(query.kwargs):
            raise ValueError(
                f"radiation.{name}.kwargs must contain DBEntry options only; "
                f"reserved loader keys: {', '.join(sorted(duplicates))}"
            )


def require_spectral_runtime(config: Config) -> None:
    """Reject configuration-only emission modes before rendering or submission."""
    if config.emitter.kind != "plasma_spectral":
        raise NotImplementedError(
            f"emitter.kind = {config.emitter.kind!r} is supported by configuration loading only. "
            "Its source builder and total-power rendering are not implemented yet."
        )


def require_runtime(config: Config) -> None:
    """Keep the published runtime restricted to the completed radiation IDS workflow."""
    if config.emitter.kind != "radiation_ids":
        raise NotImplementedError(
            "This workspace currently provides the radiation_ids runtime only"
        )


def validate_config(config: Config) -> None:
    """Validate physical/sampling constraints after parsing or CLI overrides.

    Raises
    ------
    ValueError
        If a setting is outside its supported range.
    """
    _validate_source(config)
    r, p, s = config.render, config.plasma, config.slurm
    if s.merge.cpus_per_task < 1:
        raise ValueError("slurm.merge.cpus_per_task must be positive")
    for name, value in (
        ("partition", s.partition),
        ("time", s.time),
        ("memory", s.memory),
        ("account", s.account),
        ("qos", s.qos),
        ("stdout", s.stdout),
        ("stderr", s.stderr),
        ("merge.memory", s.merge.memory),
        ("merge.time", s.merge.time),
    ):
        if value is not None and ("\n" in value or "\r" in value or "\0" in value):
            raise ValueError(f"slurm.{name} must be a single-line value")
    hdf5_entry_path(config.output.uri)
    if config.output.occurrence < 0:
        raise ValueError("output.occurrence must be nonnegative")
    sources = [("geometry.target", config.geometry.target)]
    sources.extend(
        (f"geometry.non_targets[{i}]", source)
        for i, source in enumerate(config.geometry.non_targets)
    )
    for name, source in sources:
        if source.description_index < 0 or source.occurrence < 0:
            raise ValueError(f"{name} indices must be nonnegative")
        if not math.isfinite(source.time) or source.time < 0:
            raise ValueError(f"{name}.time must be finite and nonnegative")
        if source.num_segments < 1:
            raise ValueError(f"{name}.num_segments must be positive")
    names = [source.name for source in config.geometry.non_targets]
    if len(names) != len(set(names)):
        raise ValueError("geometry.non_targets names must be unique")
    for name in ("pixel_samples", "max_passes", "save_every"):
        if getattr(r, name) < 1:
            raise ValueError(f"render.{name} must be positive")
    if r.min_samples < 2 or s.cpus_per_task < 1:
        raise ValueError("min_samples >= 2 and cpus_per_task >= 1 are required")
    if not math.isfinite(r.adaptive_cutoff) or not 0 < r.adaptive_cutoff <= 1:
        raise ValueError("render.adaptive_cutoff must be in (0, 1]")
    if not math.isfinite(r.surface_offset) or r.surface_offset < 0:
        raise ValueError("render.surface_offset must be finite and nonnegative")
    if (
        not 0 < r.adaptive_fraction <= 1
        or not math.isfinite(r.adaptive_ratio)
        or r.adaptive_ratio < 1
    ):
        raise ValueError("adaptive_fraction must be in (0, 1] and adaptive_ratio must be >= 1")
    if p is not None:
        if not math.isfinite(p.time) or (
            p.time_edge is not None and not math.isfinite(p.time_edge)
        ):
            raise ValueError("plasma times must be finite")
        if math.isnan(p.time_threshold) or p.time_threshold < 0:
            raise ValueError("plasma.time_threshold must be nonnegative (inf is allowed)")
        if p.blend_psi is not None and not (
            all(map(math.isfinite, p.blend_psi)) and p.blend_psi[0] < p.blend_psi[1]
        ):
            raise ValueError("plasma.blend_psi must contain two finite, strictly increasing values")
    if config.emitter.kind != "plasma_spectral":
        if r.wavelength_bands or r.spectral_bins is not None or r.spectral_percentile is not None:
            raise ValueError(
                "Total-power emitters do not accept render.wavelength_bands, "
                "spectral_bins or spectral_percentile; omit those settings"
            )
        return
    if r.spectral_bins is None or r.spectral_bins < 1:
        raise ValueError("render.spectral_bins must be positive")
    if r.spectral_percentile is None or not 0 < r.spectral_percentile <= 100:
        raise ValueError("render.spectral_percentile must be in (0, 100]")
    if not r.wavelength_bands:
        raise ValueError("At least one render.wavelength_bands entry is required")
    previous_upper = 0.0
    for lower, upper in r.wavelength_bands:
        if not (
            math.isfinite(lower)
            and math.isfinite(upper)
            and 0 < lower < upper
            and lower >= previous_upper
        ):
            raise ValueError(
                "render.wavelength_bands must be finite, positive, ordered and non-overlapping"
            )
        previous_upper = upper


def load_config(path: Path) -> Config:
    """Read TOML into typed dataclasses and resolve paths relative to the file.

    Returns
    -------
    Config
        Validated configuration; omitted optional values become None.

    Raises
    ------
    ValueError
        If a section, key, type or value is invalid.
    """
    path = path.expanduser().resolve()
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    classes = get_type_hints(Config)
    unknown = data.keys() - (classes.keys() - {"source"})
    if unknown:
        raise ValueError(f"Unknown configuration section(s): {', '.join(sorted(unknown))}")
    config = _section(Config, {"source": str(path), **data}, "", path.parent)
    if config.plasma is not None:
        plasma = config.plasma
        config = replace(
            config,
            plasma=replace(
                plasma,
                core_uri=_uri(plasma.core_uri, path.parent),
                edge_uri=_uri(plasma.edge_uri, path.parent) if plasma.edge_uri else None,
            ),
        )
    if config.radiation is not None:

        def resolve_query(query: RadiationInputConfig) -> RadiationInputConfig:
            kwargs = query.kwargs.copy()
            for key in ("path", "xml_path"):
                if key in kwargs:
                    kwargs[key] = str(
                        _decode(kwargs[key], Path, f"radiation kwargs.{key}", path.parent)
                    )
            return replace(query, uri=_uri(query.uri, path.parent), kwargs=kwargs)

        radiation = config.radiation
        config = replace(
            config,
            radiation=replace(
                radiation,
                primary=resolve_query(radiation.primary),
                secondary=resolve_query(radiation.secondary) if radiation.secondary else None,
            ),
        )
    geometry = config.geometry
    config = replace(
        config,
        slurm=replace(
            config.slurm,
            stdout=str(_decode(config.slurm.stdout, Path, "slurm.stdout", path.parent))
            if config.slurm.stdout
            else None,
            stderr=str(_decode(config.slurm.stderr, Path, "slurm.stderr", path.parent))
            if config.slurm.stderr
            else None,
        ),
        output=replace(config.output, uri=resolve_output_uri(config.output.uri, path.parent)),
        geometry=replace(
            geometry,
            target=replace(geometry.target, uri=_uri(geometry.target.uri, path.parent)),
            non_targets=tuple(
                replace(source, uri=_uri(source.uri, path.parent))
                for source in geometry.non_targets
            ),
        ),
    )
    validate_config(config)
    return config


def _toml_value(value: Any) -> str:
    if isinstance(value, (str, Path)):
        return json.dumps(str(value), ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(map(_toml_value, value)) + "]"
    if isinstance(value, dict):
        return (
            "{ " + ", ".join(f"{json.dumps(k)} = {_toml_value(v)}" for k, v in value.items()) + " }"
        )
    raise ValueError(f"Cannot represent {type(value).__name__} in a configuration snapshot")


def save_config(config: Config, path: Path, *, overwrite: bool = True) -> None:
    """Write effective settings as a portable TOML snapshot with absolute paths.

    Omitted optional values remain omitted, and no paths depend on the snapshot's location.
    """
    lines = ["# Effective configuration; paths are absolute."]

    def write_section(name: str, settings: Any, *, array_element: bool = False) -> None:
        header = f"[[{name}]]" if array_element else f"[{name}]"
        lines.extend(("", header))
        nested: list[tuple[str, Any, bool]] = []
        for item in fields(settings):
            value = getattr(settings, item.name)
            if value is None:
                continue
            if is_dataclass(value):
                nested.append((f"{name}.{item.name}", value, False))
            elif isinstance(value, tuple) and value and is_dataclass(value[0]):
                nested.extend((f"{name}.{item.name}", entry, True) for entry in value)
            else:
                lines.append(f"{item.name} = {_toml_value(value)}")
        for child, value, is_array in nested:
            write_section(child, value, array_element=is_array)

    for item in fields(config):
        value = getattr(config, item.name)
        if item.name != "source" and value is not None:
            write_section(item.name, value)
    with path.open("w" if overwrite else "x") as stream:
        stream.write("\n".join(lines) + "\n")
