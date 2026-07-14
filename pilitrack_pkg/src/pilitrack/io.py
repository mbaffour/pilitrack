"""Real-movie I/O for many microscope formats.

``load_movie`` loads a time-lapse from **ND2 (Nikon), TIFF / OME-TIFF, or CZI
(Zeiss)** — and ``load_array`` wraps an in-memory array — into the ``(T, Y, X)``
stacks the pipeline consumes, resolving channels (single labelled-pilus channel
vs a separate cell-body channel), projecting Z, and selecting a position/scene.
``config_from_meta`` derives an :class:`AcquisitionConfig` from the file's own
metadata (pixel size, frame interval), rescaling the pixel-unit parameters to
the real pixel size so results are comparable across rigs.

Kept out of the core so the pipeline still installs and runs on plain NumPy
arrays without any reader. ``nd2``, ``tifffile``, CZI readers and ``matplotlib``
are optional and imported lazily (mirroring the ``backends`` pattern); install
with the ``io`` and ``qc`` extras.

The pure helpers (``_to_txy``, ``_to_tcyx``, ``_dt_from_events``,
``_guess_pili_channel``, ``config_from_meta``) carry no reader dependency and are
unit-tested directly, so the loader's logic is covered without a large sample
file on hand.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import AcquisitionConfig

# Physical scale the built-in AcquisitionConfig pixel-unit defaults were tuned
# at. config_from_nd2 rescales those defaults from this to the real pixel size.
_REFERENCE_PIXEL_SIZE_NM = 65.0

# Base (unscaled) pixel-unit defaults, i.e. their values at 65 nm/px. Kept here
# so the rescale has a single source of truth rather than re-reading the config.
_BASE_RIDGE_SIGMAS = (1.0, 1.5, 2.0)
_BASE_SEARCH_RADIUS_PX = 6.0
_BASE_MAX_BASE_JUMP_PX = 5.0


def _import_nd2():
    try:
        import nd2  # noqa: WPS433 (optional dependency, imported on demand)
    except Exception as exc:  # pragma: no cover - exercised only without nd2
        raise ImportError(
            "Reading ND2 files needs the 'nd2' package. Install it with "
            "`pip install nd2` (or `pip install -e '.[io]'`)."
        ) from exc
    return nd2


# --------------------------------------------------------------------------- #
# Pure helpers (no nd2 dependency — unit-tested directly)
# --------------------------------------------------------------------------- #
def _to_txy(arr, sizes: dict, channel: int = 0, z: int = 0) -> np.ndarray:
    """Reorder an ND2 array to ``(T, Y, X)`` given its axis-size mapping.

    ``sizes`` is ``ND2File.sizes`` — an ordered dict whose keys (a subset of
    ``T, C, Z, Y, X``) match ``arr``'s axes in order. A channel is selected on
    ``C`` and a single plane on ``Z``; a missing ``T`` axis becomes a leading
    axis of length 1. Works on both NumPy and dask arrays.
    """
    axes = list(sizes.keys())
    if arr.ndim != len(axes):
        raise ValueError(f"array ndim {arr.ndim} != sizes {axes}")
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"ND2 sizes {axes} lack Y/X image axes")

    # A lone stack axis with no time axis is the time series (see _to_tcyx):
    # keep Z as the frames instead of picking a single plane.
    z_as_time = ("Z" in axes) and ("T" not in axes)

    index: list = [slice(None)] * arr.ndim
    for i, ax in enumerate(axes):
        if ax == "C":
            index[i] = int(channel)
        elif ax == "Z" and not z_as_time:
            index[i] = int(z)
    sub = arr[tuple(index)]

    dropped = ("C",) if z_as_time else ("C", "Z")
    remaining = [("T" if (ax == "Z" and z_as_time) else ax)
                 for ax in axes if ax not in dropped]
    unexpected = set(remaining) - {"T", "Y", "X"}
    if unexpected:
        raise ValueError(f"unsupported ND2 axes {sorted(unexpected)}")

    if "T" not in remaining:
        sub = sub[None, ...]
        remaining = ["T"] + remaining

    order = [remaining.index("T"), remaining.index("Y"), remaining.index("X")]
    return np.transpose(sub, order)


def _event_times(events) -> np.ndarray | None:
    """Extract the per-frame acquisition times (seconds) from ``ND2File.events``.

    Prefers the NIS ``'Time [s]'`` column; falls back to any ``time … [s]``
    key. Returns a finite 1-D array, or ``None`` if unavailable.
    """
    if not events:
        return None
    keys = list(events[0].keys())
    tkey = "Time [s]" if "Time [s]" in keys else None
    if tkey is None:
        for k in keys:
            kl = k.strip().lower()
            if kl.startswith("time") and "s]" in kl:
                tkey = k
                break
    if tkey is None:
        return None
    times = np.array([e.get(tkey, np.nan) for e in events], dtype=float)
    times = times[np.isfinite(times)]
    return times if times.size else None


def _dt_from_events(events) -> float | None:
    """Median frame-to-frame interval (seconds) from the ND2 event timestamps."""
    times = _event_times(events)
    if times is None or times.size < 2:
        return None
    return float(np.median(np.diff(times)))


def _dt_from_experiment(experiment) -> float | None:
    """Fallback: the planned period (ms) from a Time-loop in ``ND2File.experiment``."""
    if not experiment:
        return None
    for loop in experiment:
        params = getattr(loop, "parameters", None)
        periods = getattr(params, "periods", None)
        if periods:
            period_ms = getattr(periods[0], "periodMs", None)
            if period_ms:
                return float(period_ms) / 1000.0
        period_ms = getattr(params, "periodMs", None)
        if period_ms:
            return float(period_ms) / 1000.0
    return None


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def load_nd2(
    path,
    *,
    channel: int = 0,
    frames=None,
    roi=None,
    as_dask: bool = False,
) -> tuple[np.ndarray, dict]:
    """Load an ND2 movie into a ``(T, Y, X)`` array plus a metadata dict.

    Parameters
    ----------
    path : str | Path
    channel : int
        Channel index to select if the file has a ``C`` axis (single-channel
        files ignore this).
    frames : slice | sequence[int] | None
        Optional temporal subset (applied to the leading axis).
    roi : (y0, y1, x0, x1) | None
        Optional spatial crop.
    as_dask : bool
        Read lazily via ``ND2File.to_dask()`` and materialize only the selected
        frames/ROI — useful for pulling a small crop out of a large file
        without loading the whole thing.

    Returns
    -------
    stack : np.ndarray
        ``(T, Y, X)``; dtype preserved from the file (typically ``uint16``). The
        pipeline detectors cast to float per frame, so no eager float copy here.
    meta : dict
        ``pixel_size_nm, dt_s, frame_times_s, channel_names, n_timepoints,
        shape_yx, dtype, sizes, duration_s, path``. ``dt_s`` may be ``None`` if
        the file exposes no timing (pass a config override then).
    """
    nd2 = _import_nd2()
    path = Path(path)
    with nd2.ND2File(str(path)) as f:
        sizes = dict(f.sizes)
        vox = f.voxel_size()
        pixel_size_nm = float(vox.x) * 1000.0
        try:
            channel_names = [c.channel.name for c in f.metadata.channels]
        except Exception:  # pragma: no cover - metadata shape varies by file
            channel_names = []

        try:
            events = f.events()
        except Exception:  # pragma: no cover
            events = None
        frame_times = _event_times(events) if events else None
        dt_s = _dt_from_events(events) if events else None
        if dt_s is None:
            try:
                dt_s = _dt_from_experiment(f.experiment)
            except Exception:  # pragma: no cover
                dt_s = None

        arr = f.to_dask() if as_dask else f.asarray()
        stack = _to_txy(arr, sizes, channel=channel)
        if frames is not None:
            stack = stack[frames]
        if roi is not None:
            y0, y1, x0, x1 = roi
            stack = stack[:, y0:y1, x0:x1]
        if as_dask:
            stack = np.asarray(stack.compute())
        stack = np.ascontiguousarray(stack)

    if frame_times is not None and frame_times.size >= 2:
        duration_s = float(frame_times[-1] - frame_times[0])
    elif dt_s is not None:
        duration_s = float(dt_s * max(stack.shape[0] - 1, 0))
    else:
        duration_s = None

    meta = {
        "pixel_size_nm": pixel_size_nm,
        "dt_s": dt_s,
        "frame_times_s": frame_times,
        "channel_names": channel_names,
        "n_timepoints": int(stack.shape[0]),
        "shape_yx": (int(stack.shape[1]), int(stack.shape[2])),
        "dtype": str(stack.dtype),
        "sizes": sizes,
        "duration_s": duration_s,
        "path": str(path),
    }
    return stack, meta


# --------------------------------------------------------------------------- #
# Auto-config
# --------------------------------------------------------------------------- #
def config_from_meta(
    meta: dict,
    *,
    reference_pixel_size_nm: float = _REFERENCE_PIXEL_SIZE_NM,
    **overrides,
) -> AcquisitionConfig:
    """Build an :class:`AcquisitionConfig` from an ND2 ``meta`` dict.

    Sets ``dt_s``/``pixel_size_nm``/``window_s`` from the file and **rescales the
    pixel-unit params** (``ridge_sigmas``, ``base_search_radius_px``,
    ``max_base_jump_px``) so their physical meaning is preserved at the real
    pixel size instead of the 65 nm/px the defaults were tuned at. Physical (nm,
    nm/s) params keep their biological defaults, except ``velocity_sign_eps_nm_s``
    which is floored at half a pixel-step per frame — below that, pixel
    quantization masquerades as real extension/retraction. Any keyword in
    ``overrides`` wins over every computed value.
    """
    pixel_size_nm = float(meta["pixel_size_nm"])
    dt_s = meta.get("dt_s", None)
    if dt_s is None and "dt_s" not in overrides:
        raise ValueError(
            "movie metadata has no frame interval; pass dt_s=<seconds> "
            "(e.g. config_from_meta(meta, dt_s=0.4))."
        )
    dt_s = float(overrides.get("dt_s", dt_s))

    scale = reference_pixel_size_nm / pixel_size_nm
    ridge_sigmas = tuple(round(s * scale, 3) for s in _BASE_RIDGE_SIGMAS)
    velocity_eps = max(20.0, 0.5 * pixel_size_nm / dt_s) if dt_s else 20.0

    computed = dict(
        dt_s=dt_s,
        pixel_size_nm=pixel_size_nm,
        window_s=float(meta["duration_s"]) if meta.get("duration_s") else 30.0,
        ridge_sigmas=ridge_sigmas,
        base_search_radius_px=round(_BASE_SEARCH_RADIUS_PX * scale, 3),
        max_base_jump_px=round(_BASE_MAX_BASE_JUMP_PX * scale, 3),
        velocity_sign_eps_nm_s=round(velocity_eps, 3),
    )
    computed.update(overrides)
    return AcquisitionConfig(**computed)


def describe_config(cfg: AcquisitionConfig, meta: dict | None = None) -> str:
    """A short human-readable summary of the resolved acquisition config, for
    the user to eyeball before trusting the numbers (esp. dt / pixel size)."""
    lines = [
        "Resolved acquisition config:",
        f"  pixel_size_nm         = {cfg.pixel_size_nm:.3f}",
        f"  dt_s                  = {cfg.dt_s:.4f}  ({cfg.dt_s*1000:.1f} ms/frame)",
        f"  ridge_sigmas (px)     = {cfg.ridge_sigmas}",
        f"  base_search_radius_px = {cfg.base_search_radius_px}",
        f"  max_base_jump_px      = {cfg.max_base_jump_px}",
        f"  min_pilus_length_nm   = {cfg.min_pilus_length_nm} "
        f"({cfg.min_pilus_length_px:.2f} px)",
        f"  velocity_sign_eps_nm_s= {cfg.velocity_sign_eps_nm_s}",
    ]
    if meta is not None:
        lines.insert(1, f"  channels              = {meta.get('channel_names')}")
        lines.insert(1, f"  frames x shape        = "
                        f"{meta.get('n_timepoints')} x {meta.get('shape_yx')}")
    return "\n".join(lines)


# Back-compat alias: config_from_nd2 was the original ND2-specific name; the
# function works on any movie's meta dict, so config_from_meta is the general one.
config_from_nd2 = config_from_meta


# --------------------------------------------------------------------------- #
# Generic multi-format loading (ND2 / TIFF / OME-TIFF / CZI / arrays)
# --------------------------------------------------------------------------- #
# Axis letters we treat as a position/scene/series to index into (take one).
_POSITION_AXES = ("P", "S", "B", "V", "M", "R", "H")
_IMAGE_AXES = ("T", "C", "Z", "Y", "X")

# Substrings that mark a channel name as a transmitted-light / cell-body channel
# (as opposed to a fluorescence pilus channel).
_CELLISH = ("phase", "ph3", "ph2", "ph1", "bf", "bright", "trans", "dic",
            "brightfield", "tl ", "tl-", "label")
_PILISH_HINTS = ("488", "561", "640", "594", "532", "gfp", "rfp", "cy", "alexa",
                 "fitc", "tritc", "mscarlet", "mneon", "yfp", "mcherry", "nm")


def _to_tcyx(arr, sizes: dict, z="max", position: int = 0) -> np.ndarray:
    """Reorder an arbitrary microscope array to ``(T, C, Y, X)``.

    ``sizes`` maps axis letters (a subset of ``T,C,Z,Y,X`` plus a
    position/scene axis) to sizes, in ``arr``'s axis order. A position/scene
    axis is indexed to ``position``; ``Z`` is projected (``z='max'|'mean'``) or
    indexed (``z=<int>``); missing ``T``/``C`` axes are inserted as length 1.
    """
    axes = list(sizes.keys())
    if arr.ndim != len(axes):
        raise ValueError(f"array ndim {arr.ndim} != axes {axes}")
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"axes {axes} lack Y/X image axes")

    # select a single position/scene
    index = [slice(None)] * arr.ndim
    for i, ax in enumerate(axes):
        if ax in _POSITION_AXES:
            index[i] = int(position) if arr.shape[i] > position else 0
    arr = arr[tuple(index)]
    axes = [ax for ax in axes if ax not in _POSITION_AXES]

    # A lone stack axis with no explicit time axis IS the time series here:
    # TIRF pili imaging is single-plane, and plain multi-page TIFFs / ImageJ
    # stacks saved with slices=N (frames=1) report their frames on Z. Treat that
    # Z as T instead of max-projecting every frame into one. Genuine 3-D
    # time-lapses (T *and* Z present) still get Z projected below.
    if "Z" in axes and "T" not in axes:
        axes[axes.index("Z")] = "T"

    # collapse Z
    if "Z" in axes:
        zi = axes.index("Z")
        if z == "max":
            arr = np.asarray(arr).max(axis=zi)
        elif z == "mean":
            arr = np.asarray(arr).mean(axis=zi)
        else:
            zidx = [slice(None)] * arr.ndim
            zidx[zi] = int(z)
            arr = arr[tuple(zidx)]
        axes = [ax for ax in axes if ax != "Z"]

    unexpected = set(axes) - {"T", "C", "Y", "X"}
    if unexpected:
        raise ValueError(f"unsupported axes {sorted(unexpected)}")

    if "T" not in axes:
        arr = arr[None, ...]
        axes = ["T"] + axes
    if "C" not in axes:
        arr = arr[..., None]
        axes = axes + ["C"]

    order = [axes.index(k) for k in ("T", "C", "Y", "X")]
    return np.transpose(arr, order)


def _guess_pili_channel(channel_names, n_channels: int) -> int:
    """Best-guess index of the fluorescence pilus channel from channel names."""
    if n_channels == 1 or not channel_names:
        return 0
    for i, name in enumerate(channel_names[:n_channels]):
        nm = str(name).lower()
        if any(h in nm for h in _PILISH_HINTS) and not any(c in nm for c in _CELLISH):
            return i
    return 0


def _guess_cell_channel(channel_names, n_channels: int, pili_channel: int):
    """Best-guess index of a dedicated cell-body channel, else ``None``.

    Prefers a transmitted-light / phase channel by name; failing that, for a
    two-channel movie the *other* channel is used as the cell channel.
    """
    if n_channels < 2:
        return None
    if channel_names:
        for i, name in enumerate(channel_names[:n_channels]):
            if i != pili_channel and any(c in str(name).lower() for c in _CELLISH):
                return i
    if n_channels == 2:
        return 1 - pili_channel
    return None


# ---- per-format raw readers: each returns (arr_TCYX, meta_partial) ---------- #
def _read_nd2_raw(path, *, z, position, frames, roi, as_dask):
    nd2 = _import_nd2()
    with nd2.ND2File(str(path)) as f:
        sizes = dict(f.sizes)
        vox = f.voxel_size()
        pixel_size_nm = float(vox.x) * 1000.0
        try:
            channel_names = [c.channel.name for c in f.metadata.channels]
        except Exception:  # pragma: no cover
            channel_names = []
        try:
            events = f.events()
        except Exception:  # pragma: no cover
            events = None
        frame_times = _event_times(events) if events else None
        dt_s = _dt_from_events(events) if events else None
        if dt_s is None:
            try:
                dt_s = _dt_from_experiment(f.experiment)
            except Exception:  # pragma: no cover
                dt_s = None

        arr = f.to_dask() if as_dask else f.asarray()
        tcyx = _to_tcyx(arr, sizes, z=z, position=position)
        tcyx = _slice_tcyx(tcyx, frames, roi)
        tcyx = np.asarray(tcyx.compute()) if as_dask else np.asarray(tcyx)
    meta = dict(pixel_size_nm=pixel_size_nm, dt_s=dt_s,
                frame_times_s=frame_times, channel_names=channel_names,
                reader="nd2")
    return np.ascontiguousarray(tcyx), meta


def _read_tiff_raw(path, *, z, position, frames, roi, as_dask):
    try:
        import tifffile
    except Exception as exc:  # pragma: no cover
        raise ImportError("Reading TIFF needs 'tifffile' (`pip install tifffile`"
                          " or `-e '.[io]'`).") from exc
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[position] if position < len(tf.series) else tf.series[0]
        arr = series.asarray()
        axes = series.axes  # e.g. 'TCYX', 'TYX', 'TZCYX'
        sizes = dict(zip(axes, arr.shape))
        pixel_size_nm, dt_s = _tiff_scale(tf)
        channel_names = _tiff_channel_names(tf, sizes.get("C", 1))
    tcyx = _to_tcyx(arr, sizes, z=z, position=0)
    tcyx = _slice_tcyx(tcyx, frames, roi)
    meta = dict(pixel_size_nm=pixel_size_nm, dt_s=dt_s, frame_times_s=None,
                channel_names=channel_names, reader="tifffile")
    return np.ascontiguousarray(np.asarray(tcyx)), meta


def _read_czi_raw(path, *, z, position, frames, roi, as_dask):
    # Prefer aicsimageio/bioio (rich metadata); fall back to czifile.
    try:
        from aicsimageio import AICSImage  # type: ignore
    except Exception:
        AICSImage = None
    if AICSImage is not None:
        img = AICSImage(str(path))
        if position < img.dims.S if "S" in img.dims.order else 1:
            try:
                img.set_scene(position)
            except Exception:
                pass
        arr = np.asarray(img.get_image_data("TCZYX"))
        sizes = dict(zip("TCZYX", arr.shape))
        pps = img.physical_pixel_sizes
        pixel_size_nm = float(pps.X) * 1000.0 if pps and pps.X else None
        dt_s = None
        channel_names = list(img.channel_names or [])
        tcyx = _to_tcyx(arr, sizes, z=z, position=0)
    else:
        try:
            import czifile  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Reading CZI needs 'aicsimageio' or 'czifile' "
                "(`pip install aicsimageio` or `pip install czifile`)."
            ) from exc
        with czifile.CziFile(str(path)) as cz:
            arr = cz.asarray()
            axes = "".join(a for a in cz.axes)
            sizes = dict(zip(axes, arr.shape))
        pixel_size_nm, dt_s, channel_names = None, None, []
        tcyx = _to_tcyx(arr, sizes, z=z, position=position)
    tcyx = _slice_tcyx(tcyx, frames, roi)
    meta = dict(pixel_size_nm=pixel_size_nm, dt_s=dt_s, frame_times_s=None,
                channel_names=channel_names, reader="czi")
    return np.ascontiguousarray(np.asarray(tcyx)), meta


def _slice_tcyx(tcyx, frames, roi):
    if frames is not None:
        tcyx = tcyx[frames]
    if roi is not None:
        y0, y1, x0, x1 = roi
        tcyx = tcyx[:, :, y0:y1, x0:x1]
    return tcyx


def _tiff_scale(tf):
    """(pixel_size_nm, dt_s) from OME or ImageJ TIFF metadata, best-effort."""
    pixel_size_nm = None
    dt_s = None
    # OME-TIFF
    ome = getattr(tf, "ome_metadata", None)
    if ome:
        import re
        m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', ome)
        unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', ome)
        if m:
            val = float(m.group(1))
            u = (unit.group(1) if unit else "µm").lower()
            pixel_size_nm = val * (1.0 if u in ("nm", "nanometer") else 1000.0)
        t = re.search(r'TimeIncrement="([0-9.eE+-]+)"', ome)
        if t:
            dt_s = float(t.group(1))
    # ImageJ
    ij = getattr(tf, "imagej_metadata", None)
    if ij:
        if dt_s is None and ij.get("finterval"):
            dt_s = float(ij["finterval"])
    if pixel_size_nm is None:
        try:
            page = tf.pages[0]
            xres = page.tags.get("XResolution")
            runit = page.tags.get("ResolutionUnit")
            if xres is not None and xres.value[0]:
                num, den = xres.value
                per_unit = num / den  # pixels per unit
                unit_nm = 1e7 if (runit and runit.value == 3) else 2.54e7  # cm vs inch
                pixel_size_nm = unit_nm / per_unit if per_unit else None
        except Exception:  # pragma: no cover
            pixel_size_nm = None
    return pixel_size_nm, dt_s


def _tiff_channel_names(tf, n_channels):
    ome = getattr(tf, "ome_metadata", None)
    if ome:
        import re
        names = re.findall(r'<Channel[^>]*Name="([^"]+)"', ome)
        if names:
            return names[:n_channels]
    return [f"ch{i}" for i in range(int(n_channels or 1))]


_READERS = {".nd2": _read_nd2_raw, ".tif": _read_tiff_raw,
            ".tiff": _read_tiff_raw, ".czi": _read_czi_raw,
            ".ome.tif": _read_tiff_raw, ".ome.tiff": _read_tiff_raw}


def _finalize_meta(meta, tcyx, pili_channel, cell_channel, path):
    T, C, Y, X = tcyx.shape
    frame_times = meta.get("frame_times_s")
    if frame_times is not None and len(frame_times) >= 2:
        duration_s = float(frame_times[-1] - frame_times[0])
    elif meta.get("dt_s"):
        duration_s = float(meta["dt_s"] * max(T - 1, 0))
    else:
        duration_s = None
    meta.update(
        n_timepoints=int(T), n_channels=int(C), shape_yx=(int(Y), int(X)),
        dtype=str(tcyx.dtype), duration_s=duration_s,
        pili_channel=int(pili_channel),
        cell_channel=(None if cell_channel is None else int(cell_channel)),
        single_channel=(cell_channel is None), path=str(path),
    )
    return meta


def load_movie(
    path,
    *,
    pili_channel=None,
    cell_channel=None,
    z="max",
    position: int = 0,
    frames=None,
    roi=None,
    as_dask: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Load a time-lapse from any supported format into pipeline stacks.

    Dispatches by file extension: ``.nd2`` (Nikon), ``.tif``/``.tiff`` /
    ``.ome.tif`` (via ``tifffile``), ``.czi`` (via ``aicsimageio``/``czifile``).
    For in-memory arrays use :func:`load_array`.

    Returns ``(fluor_stack, cell_stack, meta)`` where ``fluor_stack`` is the
    ``(T, Y, X)`` pilus channel and ``cell_stack`` is either a *separate*
    ``(T, Y, X)`` cell-body channel or ``None`` (single-channel labelled-pilus
    movie — reuse the pilus channel via ``pilitrack.singlechannel``). Channels
    are guessed from names when not given; pass ``pili_channel`` / ``cell_channel``
    to force the roles.
    """
    suffix = "".join(Path(path).suffixes[-2:]).lower()
    reader = _READERS.get(suffix) or _READERS.get(Path(path).suffix.lower())
    if reader is None:
        raise ValueError(
            f"unsupported movie format {Path(path).suffix!r}; supported: "
            f".nd2 .tif .tiff .ome.tif .czi (or use load_array for arrays)."
        )
    tcyx, meta = reader(path, z=z, position=position, frames=frames,
                        roi=roi, as_dask=as_dask)
    C = tcyx.shape[1]
    names = meta.get("channel_names") or []
    if pili_channel is None:
        pili_channel = _guess_pili_channel(names, C)
    if cell_channel is None:
        cell_channel = _guess_cell_channel(names, C, pili_channel)

    fluor = np.ascontiguousarray(tcyx[:, int(pili_channel)])
    cell = (np.ascontiguousarray(tcyx[:, int(cell_channel)])
            if cell_channel is not None else None)
    meta = _finalize_meta(meta, tcyx, pili_channel, cell_channel, path)
    return fluor, cell, meta


def load_array(
    arr,
    *,
    axes: str = "TYX",
    pixel_size_nm=None,
    dt_s=None,
    channel_names=None,
    pili_channel=None,
    cell_channel=None,
    z="max",
    position: int = 0,
    frames=None,
    roi=None,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Wrap an in-memory array as a movie, same return contract as
    :func:`load_movie`. ``axes`` names each dimension (e.g. ``'TYX'``,
    ``'TCYX'``, ``'TZCYX'``)."""
    arr = np.asarray(arr)
    if len(axes) != arr.ndim:
        raise ValueError(f"axes {axes!r} do not match array ndim {arr.ndim}")
    sizes = dict(zip(list(axes), arr.shape))
    tcyx = _to_tcyx(arr, sizes, z=z, position=position)
    tcyx = np.ascontiguousarray(np.asarray(_slice_tcyx(tcyx, frames, roi)))
    C = tcyx.shape[1]
    names = list(channel_names) if channel_names else []
    if pili_channel is None:
        pili_channel = _guess_pili_channel(names, C)
    if cell_channel is None:
        cell_channel = _guess_cell_channel(names, C, pili_channel)
    fluor = np.ascontiguousarray(tcyx[:, int(pili_channel)])
    cell = (np.ascontiguousarray(tcyx[:, int(cell_channel)])
            if cell_channel is not None else None)
    meta = dict(pixel_size_nm=pixel_size_nm, dt_s=dt_s, frame_times_s=None,
                channel_names=names, reader="array")
    meta = _finalize_meta(meta, tcyx, pili_channel, cell_channel, path="<array>")
    return fluor, cell, meta


# --------------------------------------------------------------------------- #
# QC overlays (matplotlib optional)
# --------------------------------------------------------------------------- #
def save_qc_overlays(stack, art: dict, cfg, frames, outdir) -> list[str]:
    """Save PNG QC overlays: raw frame (gray) + detected pilus skeletons
    (magenta) + cell-label boundaries (cyan), for each index in ``frames``.

    ``art`` is a ``detect_and_link`` result. Returns the written file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "QC overlays need matplotlib. Install with `pip install matplotlib` "
            "(or `pip install -e '.[qc]'`)."
        ) from exc
    from skimage.segmentation import find_boundaries

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    per_fil = art["per_frame_filaments"]
    per_cells = art["per_frame_cell_labels"]
    H, W = art["shape"]
    written: list[str] = []

    for t in frames:
        if t < 0 or t >= stack.shape[0]:
            continue
        raw = stack[t].astype(float)
        lo, hi = np.percentile(raw, [1, 99.7])
        rgb = np.zeros((H, W, 3), dtype=float)
        norm = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)
        rgb[..., 0] = rgb[..., 1] = rgb[..., 2] = norm

        cells = per_cells[t]
        if cells is not None and np.asarray(cells).max() > 0:
            bnd = find_boundaries(np.asarray(cells), mode="outer")
            rgb[bnd] = [0.0, 1.0, 1.0]  # cyan cell edges

        for fil in per_fil[t]:
            ys = fil.coords[:, 0]
            xs = fil.coords[:, 1]
            rgb[ys, xs] = [1.0, 0.0, 1.0]  # magenta skeletons

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"frame {t}  (magenta=pili, cyan=cell edges)")
        ax.axis("off")
        fpath = outdir / f"qc_frame_{t:03d}.png"
        fig.savefig(fpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(str(fpath))
    return written
