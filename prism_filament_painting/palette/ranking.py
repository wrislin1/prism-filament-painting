"""Rank spool combinations by comparing their simulated layer colours to the image."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from prism_filament_painting.core.forward import (
    RenderConstants,
    build_forward_stack,
    even_swap_schedule,
    luminance_to_layer,
    render_preview,
    preview_sources,
    resolve_orientation,
)
from prism_filament_painting.core.prediction import calibration_catalog
from prism_filament_painting.core.metrics import banding_risk as layer_banding_risk, preview_metrics

from .color import (
    ciede2000,
    ciede2000_array,
    hex_to_rgb,
    relative_luminance,
    relative_luminance_array,
    rgb_array_to_lab,
    rgb_to_lab,
)
from .extract import extract_dominant
from .names import normalize_key
from .model import Filament, PaletteCandidate
from .scoring import LayerTargetModel, build_layer_target_model, layer_cost

# Retained for signature compatibility; no longer used by the forward-sim scorer.
DEFAULT_SAMPLE_LAYERS = (1, 2, 4, 8, 12)

# Target image is downsampled before per-pixel Lab binning to bound cost.
_TARGET_MAX_DIM = 128
_DEFAULT_LUM_BINS = 32

# Swap-schedule search: a schedule is searched per candidate (not fixed to the
# even schedule) so the layer-transition positions are tuned to the image. The
# search minimizes ``(span_misfit_to_target, fidelity, schedule)`` — see
# ``_search_placement``. It is exhaustive over strictly-increasing interior
# schedules when the count is within the cap — C(24, 3) = 2024 for palette size 4
# at the default 27/2 geometry, so sizes <= 4 are exact — else deterministic
# multi-start coordinate descent + a fixed-seed sampler (sizes 5-8).
_SCHEDULE_EXHAUSTIVE_CAP = 4096
_SCHEDULE_SAMPLES = 64  # fixed-seed diversification draws for palette sizes 5-8
_SCHEDULE_MAX_SWEEPS = 6  # coordinate-descent sweeps for the size 5-8 fidelity score

# Band placement: a pure ΔE-fidelity objective collapses the
# schedule onto the base (one translucent top filament reproduces a tone ramp with
# lower per-pixel error than several opaque bands). Instead, allocate layers to
# each filament in proportion to the image pixel mass that filament represents,
# with a visible floor for filaments the image actually uses and a 1-layer squeeze
# for filaments it does not. Fidelity then only chooses among schedules within
# ``_MISFIT_SLACK`` layers of that target allocation, so it refines color match
# near the target without being able to collapse the schedule.
_VISIBLE_BAND_LAYERS = 3  # a mass-carrying filament must occupy >= this many layers
_EMPTY_BAND_MASS = 0.03  # a filament representing < this pixel fraction is "unused"
_MISFIT_SLACK = 2  # fidelity tie-breaks among schedules within this Σ|span-target| of best

_SEARCH_PROFILES = {"alpha", "beta"}
_SCORE_PROFILES = {"alpha", "beta_perceptual", "beta_balanced"}
_BETA_RAW_CANDIDATE_CAP = 160
_BETA_EXACT_CANDIDATE_CAP = 32
_BETA_BEAM_WIDTH = 8


@dataclass(frozen=True)
class _FilamentColor:
    filament: Filament
    rgb: tuple[int, int, int]
    lab: np.ndarray
    luminance: float


@dataclass(frozen=True)
class _LumBin:
    center: float  # luminance bin center in [0, 1]
    mean_lab: np.ndarray  # pixel-count-weighted mean Lab of the bin
    weight: float  # fraction of pixels in this bin


def rank_palettes(
    image: Image.Image,
    inventory: Sequence[Filament],
    *,
    palette_size: int,
    max_candidates: int,
    order_strategy: str = "luminance_ascending",
    layer_height_mm: float = 0.08,
    sample_layers: Sequence[int] = DEFAULT_SAMPLE_LAYERS,
    render_constants: RenderConstants | None = None,
    search_profile: str = "alpha",
    score_profile: str = "alpha",
    raw_candidate_cap: int | None = None,
    exact_candidate_cap: int | None = None,
    beam_width: int | None = None,
) -> tuple[PaletteCandidate, ...]:
    if palette_size < 1:
        raise ValueError("palette_size must be at least 1")
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    if not inventory:
        raise ValueError("inventory must not be empty")
    search_profile, score_profile = _validate_scoring_profiles(
        search_profile=search_profile, score_profile=score_profile
    )
    raw_cap = raw_candidate_cap if raw_candidate_cap is not None else _BETA_RAW_CANDIDATE_CAP
    exact_cap = (
        exact_candidate_cap
        if exact_candidate_cap is not None
        else _BETA_EXACT_CANDIDATE_CAP
    )
    beam = beam_width if beam_width is not None else _BETA_BEAM_WIDTH

    constants = render_constants or RenderConstants(layer_height_mm=layer_height_mm)
    return _score_ranking_candidates(
        image,
        inventory,
        palette_size=palette_size,
        max_candidates=max_candidates,
        order_strategy=order_strategy,
        sample_layers=sample_layers,
        search_profile=search_profile,
        score_profile=score_profile,
        raw_candidate_cap=raw_cap,
        exact_candidate_cap=exact_cap,
        beam_width=beam,
        render_constants=constants,
    )


def score_palette(
    image: Image.Image,
    filaments: Sequence[Filament],
    *,
    render_constants: RenderConstants | None = None,
    search_profile: str = "alpha",
    score_profile: str = "alpha",
    beam_width: int = _BETA_BEAM_WIDTH,
) -> PaletteCandidate:
    if not filaments:
        raise ValueError("filaments must not be empty")

    constants = render_constants or RenderConstants()
    color_cache = {_filament_key(filament): _filament_color(filament) for filament in filaments}

    search_profile, score_profile = _validate_scoring_profiles(
        search_profile=search_profile, score_profile=score_profile
    )
    beta_enabled = _beta_scoring_enabled(
        search_profile=search_profile, score_profile=score_profile
    )
    if beta_enabled and beam_width < 1:
        raise ValueError("beam_width must be at least 1")

    bins = _bin_target_by_luminance(image, n_bins=_DEFAULT_LUM_BINS)
    target_model = (
        build_layer_target_model(image, constants=constants) if beta_enabled else None
    )

    return _score_fixed_palette(
        image,
        filaments,
        color_cache,
        constants,
        search_profile=search_profile,
        score_profile=score_profile,
        bins=bins,
        target_model=target_model,
        beam_width=beam_width,
    )


def _score_ranking_candidates(
    image: Image.Image,
    inventory: Sequence[Filament],
    *,
    palette_size: int,
    max_candidates: int,
    order_strategy: str,
    sample_layers: Sequence[int],
    render_constants: RenderConstants,
    search_profile: str,
    score_profile: str,
    raw_candidate_cap: int,
    exact_candidate_cap: int,
    beam_width: int,
) -> tuple[PaletteCandidate, ...]:
    del sample_layers  # retained for compatibility with rank_palettes' prior signature
    constants = render_constants
    inventory = _dedupe_by_filament_name(inventory)
    target_size = min(palette_size, len(inventory))
    color_cache = {
        _filament_key(filament): _filament_color(filament) for filament in inventory
    }
    beta_enabled = _beta_scoring_enabled(
        search_profile=search_profile, score_profile=score_profile
    )

    if beta_enabled:
        if raw_candidate_cap < 1:
            raise ValueError("raw_candidate_cap must be at least 1")
        if exact_candidate_cap < 1:
            raise ValueError("exact_candidate_cap must be at least 1")
        if beam_width < 1:
            raise ValueError("beam_width must be at least 1")

    bins = _bin_target_by_luminance(image, n_bins=_DEFAULT_LUM_BINS)
    target_model = (
        build_layer_target_model(image, constants=constants) if beta_enabled else None
    )

    candidate_sets, generation = _candidate_sets_with_diagnostics(
        image,
        inventory,
        color_cache,
        target_size,
        search_profile=search_profile,
        raw_candidate_cap=raw_candidate_cap,
    )
    candidates: dict[tuple, tuple[Filament, ...]] = {}
    for selected in candidate_sets:
        if not selected:
            continue
        ordered = _order_filaments(selected, order_strategy, color_cache)
        candidates[_candidate_key(ordered)] = ordered

    candidate_values = list(candidates.values())
    generation["raw_considered"] = len(candidate_values)
    if beta_enabled and len(candidate_values) > exact_candidate_cap:
        prefit_dominant = _candidate_prefit_dominants(image, target_size)
        scored_for_prune = [
            (_candidate_prefit_score(filaments, color_cache, prefit_dominant), filaments)
            for filaments in candidate_values
        ]
        scored_for_prune.sort(key=lambda item: (item[0], _candidate_key(item[1])))
        candidate_values = [filaments for _score, filaments in scored_for_prune[:exact_candidate_cap]]
    generation["exact_scored"] = len(candidate_values)
    generation["dropped_before_exact"] = max(
        0, generation["raw_considered"] - generation["exact_scored"]
    )

    ranked = []
    for filaments in candidate_values:
        candidate = _score_fixed_palette(
            image,
            filaments,
            color_cache,
            constants,
            search_profile=search_profile,
            score_profile=score_profile,
            beam_width=beam_width,
            bins=bins,
            target_model=target_model,
            candidate_generation=generation,
        )
        ranked.append(candidate)
    ranked.sort(key=lambda candidate: (candidate.score, _candidate_key(candidate.filaments)))
    return tuple(ranked[:max_candidates])


def _validate_scoring_profiles(search_profile: str, score_profile: str) -> tuple[str, str]:
    if search_profile not in _SEARCH_PROFILES:
        raise ValueError(f"search_profile must be one of {sorted(_SEARCH_PROFILES)}")
    if score_profile not in _SCORE_PROFILES:
        raise ValueError(f"score_profile must be one of {sorted(_SCORE_PROFILES)}")
    if search_profile == "beta" and score_profile == "alpha":
        score_profile = "beta_balanced"
    return search_profile, score_profile


def _beta_scoring_enabled(search_profile: str, score_profile: str) -> bool:
    return search_profile == "beta" or score_profile != "alpha"


def _score_fixed_palette(
    image: Image.Image,
    filaments: Sequence[Filament],
    color_cache: dict,
    constants: RenderConstants,
    *,
    search_profile: str,
    score_profile: str,
    beam_width: int,
    bins: Sequence[_LumBin] | None = None,
    target_model: LayerTargetModel | None = None,
    candidate_generation: dict[str, int] | None = None,
    placement_only: bool = False,
) -> PaletteCandidate:
    beta_enabled = _beta_scoring_enabled(
        search_profile=search_profile, score_profile=score_profile
    )
    if beta_enabled:
        if target_model is None:
            target_model = build_layer_target_model(image, constants=constants)
        if bins is None:
            bins = _bin_target_by_luminance(image, n_bins=_DEFAULT_LUM_BINS)
    else:
        bins = _bin_target_by_luminance(image, n_bins=_DEFAULT_LUM_BINS) if bins is None else bins

    if score_profile == "alpha":
        schedule, score, diagnostics = _score_candidate(
            filaments, color_cache, bins, constants, placement_only=placement_only
        )
    else:
        assert target_model is not None
        schedule, score, diagnostics = _score_candidate_beta(
            filaments,
            color_cache,
            target_model,
            constants,
            image,
            score_profile=score_profile,
            search_profile=search_profile,
            beam_width=beam_width,
            placement_only=placement_only,
        )

    if beta_enabled:
        diagnostics = {
            **diagnostics,
            "search_profile": search_profile,
            "score_profile": score_profile,
        }
        if candidate_generation is not None:
            diagnostics["candidate_generation"] = dict(candidate_generation)

    return PaletteCandidate(
        filaments=tuple(filaments),
        score=score,
        swap_schedule=schedule,
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# Target representation: luminance-stratified bins                            #
# --------------------------------------------------------------------------- #


def _bin_target_by_luminance(image: Image.Image, *, n_bins: int) -> tuple[_LumBin, ...]:
    if "A" in image.getbands():
        # Transparent pixels are auto-flattened to the base filament by the renderer
        # regardless of palette, so they are background, not art to optimize.
        # Exclude alpha==0 rather than mis-binning their leftover RGB as bright.
        rgba = image if image.mode == "RGBA" else image.convert("RGBA")
        if max(rgba.size) > _TARGET_MAX_DIM:
            rgba = rgba.copy()
            # NEAREST keeps alpha discrete (no edge-blended partial alpha).
            rgba.thumbnail((_TARGET_MAX_DIM, _TARGET_MAX_DIM), Image.NEAREST)
        rgba_arr = np.asarray(rgba, dtype=np.uint8).reshape(-1, 4)
        arr = rgba_arr[:, :3]
        opaque = rgba_arr[:, 3] != 0
        if opaque.any():  # guard: fully-transparent image -> score all pixels
            arr = arr[opaque]
    else:
        rgb_image = image.convert("RGB")
        if max(rgb_image.size) > _TARGET_MAX_DIM:
            rgb_image = rgb_image.copy()
            rgb_image.thumbnail((_TARGET_MAX_DIM, _TARGET_MAX_DIM))
        arr = np.asarray(rgb_image, dtype=np.uint8).reshape(-1, 3)
    if arr.shape[0] == 0:
        raise ValueError("image has no pixels")
    lum = relative_luminance_array(arr)
    lab = rgb_array_to_lab(arr)
    index = np.clip((lum * n_bins).astype(int), 0, n_bins - 1)
    total = float(arr.shape[0])
    bins: list[_LumBin] = []
    for bucket in range(n_bins):
        mask = index == bucket
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            _LumBin(
                center=(bucket + 0.5) / n_bins,
                mean_lab=lab[mask].mean(axis=0),
                weight=count / total,
            )
        )
    return tuple(bins)


# --------------------------------------------------------------------------- #
# Candidate generation: dominant colors + tonal anchors + TD spread           #
# --------------------------------------------------------------------------- #


def _candidate_sets(
    image: Image.Image,
    inventory: Sequence[Filament],
    color_cache: dict,
    target_size: int,
) -> list[tuple[Filament, ...]]:
    dominant = [
        (rgb_to_lab(rgb), float(weight))
        for rgb, weight in extract_dominant(image, k=target_size)
    ]
    nearest_by_color = [
        _nearest_filaments(lab, inventory, color_cache, limit=min(2, len(inventory)))
        for lab, _weight in dominant
    ]
    global_fill = _global_nearest(dominant, inventory, color_cache)

    sets: list[tuple[Filament, ...]] = []
    for combo in itertools.product(*nearest_by_color):
        sets.append(_fill_unique(combo, global_fill, target_size))
    sets.append(_tonal_anchor_set(inventory, color_cache, target_size, global_fill))
    sets.append(_td_spread_set(global_fill, target_size))
    return sets


def _candidate_sets_with_diagnostics(
    image: Image.Image,
    inventory: Sequence[Filament],
    color_cache: dict,
    target_size: int,
    *,
    search_profile: str,
    raw_candidate_cap: int,
) -> tuple[list[tuple[Filament, ...]], dict]:
    if search_profile == "alpha":
        sets = _candidate_sets(image, inventory, color_cache, target_size)
        return sets, {
            "profile": "alpha",
            "raw_generated": len(sets),
            "raw_cap": len(sets),
            "raw_dropped": 0,
        }
    sets, dropped = _candidate_sets_beta(
        image,
        inventory,
        color_cache,
        target_size,
        raw_candidate_cap=raw_candidate_cap,
    )
    return sets, {
        "profile": "beta",
        "raw_generated": len(sets) + dropped,
        "raw_cap": raw_candidate_cap,
        "raw_dropped": dropped,
    }


def _candidate_sets_beta(
    image: Image.Image,
    inventory: Sequence[Filament],
    color_cache: dict,
    target_size: int,
    *,
    raw_candidate_cap: int,
) -> tuple[list[tuple[Filament, ...]], int]:
    dominant = [
        (rgb_to_lab(rgb), float(weight))
        for rgb, weight in extract_dominant(image, k=min(max(target_size * 2, target_size), 12))
    ]
    nearest_limit = min(3, len(inventory))
    nearest_by_color = [
        _nearest_filaments(lab, inventory, color_cache, limit=nearest_limit)
        for lab, _weight in dominant[: max(1, target_size)]
    ]
    global_fill = _global_nearest(dominant, inventory, color_cache)
    tonal = _tonal_anchor_set(inventory, color_cache, target_size, global_fill)
    td_spread = _td_spread_set(global_fill, target_size)
    bridge = _neutral_bridge_set(inventory, color_cache, target_size, global_fill)

    sets: list[tuple[Filament, ...]] = []
    seen: set[tuple] = set()

    def add(selected: Sequence[Filament]) -> None:
        filled = _fill_unique(selected, global_fill, target_size)
        key = _candidate_key(filled)
        if key in seen:
            return
        seen.add(key)
        if len(sets) >= raw_candidate_cap:
            return
        sets.append(filled)

    for seed in (tonal, td_spread, bridge, global_fill[:target_size]):
        add(seed)
    if nearest_by_color:
        for combo in itertools.product(*nearest_by_color):
            add(combo)
    # Add single bridge swaps around the globally nearest set; these often cover
    # neutral midtones that are not dominant enough to win nearest-color search.
    for filament in bridge:
        add((filament, *global_fill))
    dropped = max(0, len(seen) - len(sets))
    return sets, dropped


def _neutral_bridge_set(
    inventory: Sequence[Filament],
    color_cache: dict,
    target_size: int,
    global_fill: Sequence[Filament],
) -> tuple[Filament, ...]:
    scored = []
    for filament in inventory:
        color = color_cache[_filament_key(filament)]
        chroma = float(np.hypot(color.lab[1], color.lab[2]))
        mid_l = abs(float(color.lab[0]) - 55.0)
        scored.append((chroma + mid_l * 0.25, filament))
    scored.sort(key=lambda item: (item[0], item[1].brand, item[1].name))
    return _fill_unique(
        [filament for _score, filament in scored[: max(1, target_size // 2)]],
        global_fill,
        target_size,
    )


def _candidate_prefit_dominants(
    image: Image.Image,
    target_size: int,
) -> list[tuple[np.ndarray, float]]:
    return [
        (rgb_to_lab(rgb), float(weight))
        for rgb, weight in extract_dominant(image, k=min(max(target_size, 1), 8))
    ]


def _candidate_prefit_score(
    filaments: Sequence[Filament],
    color_cache: dict,
    dominant: Sequence[tuple[np.ndarray, float]],
) -> float:
    if not dominant:
        return 0.0
    filament_labs = np.stack([color_cache[_filament_key(f)].lab for f in filaments])
    total = 0.0
    for lab, weight in dominant:
        repeated = np.repeat(np.asarray(lab).reshape(1, 3), filament_labs.shape[0], axis=0)
        total += float(weight) * float(np.min(ciede2000_array(repeated, filament_labs)))
    td_values = [float(f.td) for f in filaments]
    td_span = max(td_values) - min(td_values) if td_values else 0.0
    return total - min(td_span, 4.0) * 0.05


def _nearest_filaments(
    lab: np.ndarray,
    inventory: Sequence[Filament],
    color_cache: dict,
    *,
    limit: int,
) -> tuple[Filament, ...]:
    scored = [
        (ciede2000(lab, color_cache[_filament_key(filament)].lab), filament)
        for filament in inventory
    ]
    scored.sort(key=lambda item: (item[0], item[1].brand, item[1].name))
    return tuple(filament for _score, filament in scored[:limit])


def _global_nearest(
    dominant: Sequence[tuple[np.ndarray, float]],
    inventory: Sequence[Filament],
    color_cache: dict,
) -> tuple[Filament, ...]:
    scored = []
    for filament in inventory:
        filament_lab = color_cache[_filament_key(filament)].lab
        error = sum(weight * ciede2000(lab, filament_lab) for lab, weight in dominant)
        scored.append((error, filament))
    scored.sort(key=lambda item: (item[0], item[1].brand, item[1].name))
    return tuple(filament for _score, filament in scored)


def _tonal_anchor_set(
    inventory: Sequence[Filament],
    color_cache: dict,
    target_size: int,
    global_fill: Sequence[Filament],
) -> tuple[Filament, ...]:
    """Guarantee dark + light + mid-neutral coverage, then fill by color relevance."""
    by_lum = sorted(
        inventory,
        key=lambda f: (color_cache[_filament_key(f)].luminance, f.brand, f.name),
    )
    chosen: list[Filament] = []
    seen: set = set()

    def add(filament: Filament) -> None:
        key = _filament_key(filament)
        if key not in seen and len(chosen) < target_size:
            seen.add(key)
            chosen.append(filament)

    if by_lum:
        add(by_lum[0])  # darkest
        add(by_lum[-1])  # lightest
    mids = sorted(
        inventory,
        key=lambda f: (
            abs(color_cache[_filament_key(f)].luminance - 0.5),
            f.brand,
            f.name,
        ),
    )
    for filament in mids:
        lab = color_cache[_filament_key(filament)].lab
        if math.hypot(float(lab[1]), float(lab[2])) < 18.0:  # near-neutral
            add(filament)
            break
    for filament in itertools.chain(global_fill, by_lum):
        add(filament)
    return tuple(chosen)


def _td_spread_set(
    global_fill: Sequence[Filament], target_size: int
) -> tuple[Filament, ...]:
    """A color-relevant set chosen to span low->high TD."""
    pool = list(global_fill[: max(target_size * 3, target_size)])
    pool.sort(key=lambda f: (f.td, f.brand, f.name))
    if len(pool) <= target_size:
        return tuple(pool)
    if target_size == 1:
        return (pool[0],)
    picks = [round(i * (len(pool) - 1) / (target_size - 1)) for i in range(target_size)]
    return tuple(pool[index] for index in dict.fromkeys(picks))


# --------------------------------------------------------------------------- #
# Scoring: forward-sim render fidelity + diagnostics                          #
# --------------------------------------------------------------------------- #


def _score_candidate(
    filaments: Sequence[Filament],
    color_cache: dict,
    bins: Sequence[_LumBin],
    constants: RenderConstants,
    *,
    placement_only: bool = False,
) -> tuple[tuple[int, ...], float, dict]:
    score, mass, target = _make_score_fn(filaments, color_cache, bins, constants)

    # Place transitions so each filament's layer share tracks the image pixel mass
    # it represents (anti-collapse), then let fidelity refine near that target.
    schedule, objective, diagnostics = _search_placement(
        len(filaments), constants, target, mass, score, placement_only=placement_only
    )
    diagnostics = {
        **diagnostics,
        **_placement_diagnostics(schedule, mass, constants),
    }
    return schedule, objective, diagnostics


def _make_score_fn(
    filaments: Sequence[Filament],
    color_cache: dict,
    bins: Sequence[_LumBin],
    constants: RenderConstants,
):
    bin_centers = np.array([b.center for b in bins], dtype=float)
    bin_labs = np.stack([b.mean_lab for b in bins])
    bin_weights = np.array([b.weight for b in bins], dtype=float)
    filament_labs = np.stack([color_cache[_filament_key(f)].lab for f in filaments])

    # The bin -> layer_index mapping depends only on orientation (+ geometry), not
    # the schedule, so precompute it for both orientations once. Orientation itself
    # IS schedule-dependent (the top-of-stack color shifts with transition
    # positions), so it is resolved per schedule inside ``score`` below.
    layer_index_by_orientation = {
        orientation: luminance_to_layer(
            bin_centers, constants=constants, orientation=orientation
        )
        for orientation in ("bright", "dark")
    }

    def score(schedule: tuple[int, ...]) -> tuple[float, dict]:
        stack = build_forward_stack(
            filaments, constants=constants, swap_schedule=schedule
        )
        orientation = resolve_orientation(stack, orientation="auto")
        layer_index = layer_index_by_orientation[orientation]
        pred_rgb = stack.layer_rgb[layer_index]
        pred_lab = rgb_array_to_lab(pred_rgb)
        delta_e = ciede2000_array(pred_lab, bin_labs)
        fidelity = float(np.sum(bin_weights * delta_e))
        diagnostics = _diagnostics(bins, bin_labs, pred_lab, filament_labs, constants)
        if calibration_catalog():
            diagnostics["prediction_sources_by_layer"] = stack.layer_sources
            diagnostics["prediction_sources"] = sorted({stack.layer_sources[i] for i in layer_index})
        return fidelity + _penalties(diagnostics), diagnostics

    # Place transitions so each filament's layer share tracks the image pixel mass
    # it represents (anti-collapse), then let fidelity refine near that target.
    mass, target = _tonal_allocation(filaments, color_cache, bins, constants)
    return score, mass, target


def _score_candidate_beta(
    filaments: Sequence[Filament],
    color_cache: dict,
    target_model: LayerTargetModel,
    constants: RenderConstants,
    image: Image.Image,
    *,
    score_profile: str,
    search_profile: str,
    beam_width: int,
    placement_only: bool = False,
) -> tuple[tuple[int, ...], float, dict]:
    score, mass, target = _make_beta_score_fn(
        filaments, color_cache, target_model, constants, score_profile=score_profile
    )
    if search_profile == "beta" and len(filaments) > 4:
        schedule, objective, diagnostics = _beam_search_placement(
            len(filaments), constants, target, mass, score, beam_width=beam_width
        )
    else:
        schedule, objective, diagnostics = _search_placement(
            len(filaments), constants, target, mass, score, placement_only=placement_only
        )
        diagnostics = {"schedule_search": "exhaustive", **diagnostics}
    diagnostics = {
        **diagnostics,
        **_placement_diagnostics(schedule, mass, constants),
    }
    if placement_only:
        return schedule, objective, diagnostics
    stack = build_forward_stack(filaments, constants=constants, swap_schedule=schedule)
    orientation = resolve_orientation(stack, orientation="auto")
    diagnostics["preview_metrics"] = _preview_metrics_for_candidate(
        image, stack, orientation
    )
    return schedule, objective, diagnostics


def _make_beta_score_fn(
    filaments: Sequence[Filament],
    color_cache: dict,
    target_model: LayerTargetModel,
    constants: RenderConstants,
    *,
    score_profile: str,
):
    filament_labs = np.stack([color_cache[_filament_key(f)].lab for f in filaments])
    filament_lums = np.array(
        [color_cache[_filament_key(f)].luminance for f in filaments], dtype=float
    )

    def score(schedule: tuple[int, ...]) -> tuple[float, dict]:
        stack = build_forward_stack(filaments, constants=constants, swap_schedule=schedule)
        orientation = resolve_orientation(stack, orientation="auto")
        targets = target_model.by_orientation[orientation]
        pred_lab_by_layer = rgb_array_to_lab(stack.layer_rgb)
        if not targets:
            fidelity = 0.0
        else:
            fidelity = float(
                sum(
                    target.weight * layer_cost(pred_lab_by_layer[target.layer], target)
                    for target in targets
                )
            )
        diagnostics = _beta_diagnostics(targets, pred_lab_by_layer, filament_labs, constants)
        diagnostics.update(layer_banding_risk(stack.layer_rgb))
        if calibration_catalog():
            diagnostics["prediction_sources_by_layer"] = stack.layer_sources
            diagnostics["prediction_sources"] = sorted({stack.layer_sources[t.layer] for t in targets})
        if score_profile == "beta_balanced":
            fidelity += _penalties(diagnostics) + diagnostics.get("banding_risk", 0.0) * 1.5
        return fidelity, diagnostics

    bins = _lum_bins_from_layer_targets(target_model.by_orientation["bright"], constants)
    mass = _filament_tonal_mass(filament_lums, bins) if bins else np.ones(len(filaments))
    if float(np.sum(mass)) > 0:
        mass = mass / float(np.sum(mass))
    target = _target_layer_allocation(mass, len(filaments), constants)
    return score, mass, target


def _beta_diagnostics(
    targets,
    pred_lab_by_layer: np.ndarray,
    filament_labs: np.ndarray,
    constants: RenderConstants,
) -> dict:
    if targets:
        target_l = np.array([target.mean_lab[0] for target in targets], dtype=float)
        pred_lab = np.stack([pred_lab_by_layer[target.layer] for target in targets])
        pred_l = pred_lab[:, 0]
        target_span = float(target_l.max() - target_l.min())
        pred_span = float(pred_l.max() - pred_l.min())
        coverage = pred_span / target_span if target_span > 1e-6 else 1.0
        distribution_cost = float(
            sum(target.weight * layer_cost(pred_lab_by_layer[target.layer], target) for target in targets)
        )
        chroma_collision = float(sum(target.weight * target.chroma_variance for target in targets))
    else:
        pred_lab = pred_lab_by_layer
        coverage = 1.0
        distribution_cost = 0.0
        chroma_collision = 0.0

    if pred_lab.shape[0] > 1:
        adjacent = ciede2000_array(pred_lab[:-1], pred_lab[1:])
        min_contrast = float(adjacent.min())
    else:
        min_contrast = 99.0

    chroma = np.hypot(filament_labs[:, 1], filament_labs[:, 2])
    filament_l = filament_labs[:, 0]
    has_mid_neutral = bool(
        np.any((chroma < 12.0) & (filament_l > 30.0) & (filament_l < 75.0))
    )
    mid_weight = float(
        sum(target.weight for target in targets if constants.base_layers < target.layer < constants.total_layers)
    )
    missing_midtone = (mid_weight > 0.15) and not has_mid_neutral
    printable_layers = constants.total_layers - constants.base_layers
    layer_starved = printable_layers < max(1, len(targets)) * 0.5
    return {
        "tonal_coverage": round(coverage, 3),
        "weak_tonal_coverage": coverage < 0.7,
        "min_band_contrast": round(min_contrast, 2),
        "low_contrast": min_contrast < 5.0,
        "missing_midtone_neutral": missing_midtone,
        "layer_starved_gradient": layer_starved,
        "distribution_delta_e": round(distribution_cost, 3),
        "chroma_collision": round(chroma_collision, 3),
    }


def _preview_metrics_for_candidate(image: Image.Image, stack, orientation: str) -> dict:
    preview = render_preview(image, stack, orientation=orientation)
    target = image.convert("RGB")
    if target.size != preview.size:
        target = target.resize(preview.size)
    transparent = _transparent_mask(image, preview.size)
    metrics = preview_metrics(
        np.asarray(preview, dtype=np.uint8),
        np.asarray(target, dtype=np.uint8),
        transparent_mask=transparent,
        layer_rgb=stack.layer_rgb,
    )
    if calibration_catalog():
        metrics["prediction_sources"] = preview_sources(image, stack, transparent)
    return metrics


def _transparent_mask(image: Image.Image, size: tuple[int, int]) -> np.ndarray | None:
    if "A" not in image.getbands():
        return None
    alpha = image.getchannel("A")
    if alpha.size != size:
        alpha = alpha.resize(size)
    return np.asarray(alpha, dtype=np.uint8) == 0


def _lum_bins_from_layer_targets(targets, constants: RenderConstants) -> tuple[_LumBin, ...]:
    printable = max(1, constants.total_layers - constants.base_layers)
    bins = []
    for target in targets:
        center = (target.layer - constants.base_layers) / printable
        bins.append(
            _LumBin(
                center=float(np.clip(center, 0.0, 1.0)),
                mean_lab=target.mean_lab,
                weight=target.weight,
            )
        )
    return tuple(bins)


def _tonal_allocation(filaments, color_cache, bins, constants):
    luminances = np.array([color_cache[_filament_key(f)].luminance for f in filaments])
    mass = _filament_tonal_mass(luminances, bins)
    return mass, _target_layer_allocation(mass, len(filaments), constants)


def _filament_tonal_mass(
    filament_lums: np.ndarray, bins: Sequence[_LumBin]
) -> np.ndarray:
    """Schedule-independent per-filament pixel mass.

    Each luminance bin is assigned to the nearest filament *by filament luminance*
    (ties -> lowest index), and its pixel-fraction weight is summed there. Answers
    "how much of the image does this filament represent" without reference to any
    swap schedule, so it is a stable target for placement.
    """
    centers = np.array([b.center for b in bins], dtype=float)
    weights = np.array([b.weight for b in bins], dtype=float)
    nearest = np.abs(centers[:, None] - filament_lums[None, :]).argmin(axis=1)
    mass = np.zeros(len(filament_lums), dtype=float)
    for j in range(len(filament_lums)):
        mass[j] = float(weights[nearest == j].sum())
    return mass


def _band_floor(printable: int, palette_size: int) -> int:
    """Min layers a mass-carrying filament must occupy to read as its own band.

    ``_VISIBLE_BAND_LAYERS`` normally; adapts down for degenerate geometry so the
    floor is always satisfiable. Single source of truth shared by the allocation,
    the placement floor filter, and the collapse diagnostic.
    """
    return max(1, min(_VISIBLE_BAND_LAYERS, printable // palette_size))


def _target_layer_allocation(
    mass: np.ndarray, palette_size: int, constants: RenderConstants
) -> tuple[int, ...]:
    """Target layer count per filament: ∝ mass, floored for massful, 1 for unused.

    Filaments carrying real pixel mass (> ``_EMPTY_BAND_MASS``) get layers in
    proportion to that mass with a ``_band_floor`` floor; filaments the image does
    not use get a single layer so their layers go to the ones it does. Integer
    counts sum exactly to the printable layer count (largest-remainder rounding).
    The floor adapts down for degenerate geometry so it always fits.
    """
    printable = constants.total_layers - constants.base_layers
    floor = _band_floor(printable, palette_size)
    massful = mass > _EMPTY_BAND_MASS
    target = np.ones(palette_size, dtype=int)  # unused filaments -> 1 layer
    idx = np.where(massful)[0]
    if len(idx) == 0:  # no mass anywhere (degenerate) -> even spacing
        return _even_swap_layer_counts(palette_size, printable)
    pool = printable - (palette_size - len(idx))  # layers for the massful filaments
    base_each = min(floor, pool // len(idx))
    remaining = pool - base_each * len(idx)
    share = mass[idx] / mass[idx].sum()
    extra_float = share * remaining
    extra = np.floor(extra_float).astype(int)
    short = remaining - int(extra.sum())
    for i in np.argsort(-(extra_float - extra))[:short]:
        extra[i] += 1
    target[idx] = base_each + extra
    return tuple(int(x) for x in target)


def _even_swap_layer_counts(palette_size: int, printable: int) -> tuple[int, ...]:
    """Even integer layer counts summing to ``printable`` (degenerate fallback)."""
    base_each = printable // palette_size
    counts = [base_each] * palette_size
    for i in range(printable - base_each * palette_size):
        counts[i] += 1
    return tuple(counts)


def _target_schedule(target: Sequence[int], base: int) -> tuple[int, ...]:
    """The unique schedule whose band spans equal ``target`` (cumulative sum)."""
    schedule: list[int] = []
    cum = base
    for count in target[:-1]:
        cum += int(count)
        schedule.append(cum)
    return tuple(schedule)


def _spans(
    schedule: Sequence[int], *, palette_size: int, base: int, total: int
) -> tuple[int, ...]:
    bounds = (base, *schedule, total)
    return tuple(int(bounds[i + 1] - bounds[i]) for i in range(palette_size))


def _misfit(spans: Sequence[int], target: Sequence[int]) -> int:
    return int(sum(abs(spans[i] - target[i]) for i in range(len(target))))


def _collapses(spans: Sequence[int], mass: np.ndarray, *, floor: int) -> bool:
    return any(spans[i] < floor for i in range(len(mass)) if mass[i] > _EMPTY_BAND_MASS)


@lru_cache(maxsize=512)
def _eligible_placements(
    base: int, total: int, target: tuple[int, ...], floors: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """Mass-eligible schedules depend only on geometry and integer layer spans."""
    size = len(target)
    if math.comb(total - base - 1, size - 1) <= _SCHEDULE_EXHAUSTIVE_CAP:
        candidates = itertools.combinations(range(base + 1, total), size - 1)
    else:
        candidates = _placement_candidates(size, base, total, target)
    # The target itself is in either search space, so the minimum misfit is zero.
    eligible = []
    for schedule in candidates:
        spans = _spans(schedule, palette_size=size, base=base, total=total)
        if _misfit(spans, target) <= _MISFIT_SLACK and all(
            n >= floor for n, floor in zip(spans, floors)
        ):
            eligible.append(schedule)
    return tuple(eligible) or (_target_schedule(target, base),)


def _search_placement(
    palette_size: int,
    constants: RenderConstants,
    target: Sequence[int],
    mass: np.ndarray,
    score,
    *,
    placement_only: bool = False,
) -> tuple[tuple[int, ...], float, dict]:
    """Pick the schedule whose spans best match ``target``; fidelity breaks ties.

    Placement minimizes ``(span_misfit, fidelity, schedule)`` where ``span_misfit``
    is ``Σ|span_i - target_i|`` and fidelity (the ΔE objective from
    :func:`_score_candidate`) only chooses among schedules within ``_MISFIT_SLACK``
    of the best misfit **and never below the visible band floor for a mass-carrying
    filament** — the slack alone could otherwise shrink a floored band by up to
    ``_MISFIT_SLACK`` layers and re-collapse it, so the floor is a hard filter here.

    The **returned objective is the palette's best achievable fidelity** (min over
    schedules), kept deliberately separate from the placed schedule so
    candidate ranking uses the same fidelity objective for every palette size: exhaustive for size <= 4, multi-start coordinate descent
    (:func:`_min_fidelity_descent`) for 5-8. Consequence (intentional): one
    ``PaletteCandidate`` carries ``.score`` = best-achievable fidelity (ranking /
    palette quality) while ``.swap_schedule`` + ``.diagnostics`` describe the
    *mass-placed* schedule that is actually printed — these two intentionally
    describe different schedules. In placement-only mode the objective describes
    the placed schedule and is used only during candidate selection.
    """
    base = constants.base_layers
    total = constants.total_layers
    transitions = palette_size - 1
    if transitions <= 0:
        objective, diagnostics = score(())
        return (), objective, diagnostics

    positions = list(range(base + 1, total))
    if len(positions) < transitions:
        # Genuinely infeasible: fewer interior layers than transitions needed, so no
        # strictly-increasing schedule exists. Fail loud — only reachable via custom
        # render_constants (Prism's HTTP path is always 27/2).
        raise ValueError(
            f"not enough printable layers for palette size {palette_size}"
        )

    floor = _band_floor(total - base, palette_size)
    floors = tuple(floor if weight > _EMPTY_BAND_MASS else 1 for weight in mass)
    placed = {}
    for schedule in _eligible_placements(base, total, tuple(target), floors):
        placed[schedule] = score(schedule)
    schedule = min(placed, key=lambda s: (placed[s][0], s))
    placed_objective, diagnostics = placed[schedule]
    # Suggestions compare rendered placements. Defer the independent best-achievable
    # fidelity search until a set actually needs a published rank score.
    if placement_only:
        return schedule, placed_objective, diagnostics
    if math.comb(len(positions), transitions) <= _SCHEDULE_EXHAUSTIVE_CAP:
        best_fidelity = min(
            placed[s][0] if s in placed else score(s)[0]
            for s in itertools.combinations(positions, transitions)
        )
    else:
        best_fidelity = _min_fidelity_descent(palette_size, base, total, score)
    return schedule, best_fidelity, diagnostics


def _beam_search_placement(
    palette_size: int,
    constants: RenderConstants,
    target: Sequence[int],
    mass: np.ndarray,
    score,
    *,
    beam_width: int,
) -> tuple[tuple[int, ...], float, dict]:
    base = constants.base_layers
    total = constants.total_layers
    transitions = palette_size - 1
    if transitions <= 0:
        objective, diagnostics = score(())
        return (), objective, {"schedule_search": "beam", **diagnostics}

    printable = total - base
    floor = _band_floor(printable, palette_size)
    min_spans = _minimum_spans(mass, floor=floor, printable=printable)
    if sum(min_spans) > printable:
        min_spans = (1,) * palette_size

    beam: list[tuple[int, ...]] = [()]
    scored_seen: dict[tuple[int, ...], tuple[float, dict]] = {}

    def scored(schedule: tuple[int, ...]) -> tuple[float, dict]:
        if schedule not in scored_seen:
            scored_seen[schedule] = score(schedule)
        return scored_seen[schedule]

    for step in range(transitions):
        expanded: list[tuple[float, float, tuple[int, ...], tuple[int, ...]]] = []
        for partial in beam:
            previous = partial[-1] if partial else base
            min_current = min_spans[step]
            min_remaining = sum(min_spans[step + 1 :])
            first = previous + min_current
            last = total - min_remaining
            for layer in range(first, last + 1):
                next_partial = (*partial, layer)
                completed = _complete_schedule_from_partial(
                    next_partial,
                    palette_size=palette_size,
                    base=base,
                    total=total,
                    min_spans=min_spans,
                )
                spans = _spans(completed, palette_size=palette_size, base=base, total=total)
                if _collapses(spans, mass, floor=floor):
                    continue
                objective, _diagnostics = scored(completed)
                expanded.append((_misfit(spans, target), objective, completed, next_partial))
        if not expanded:
            fallback = _target_schedule(target, base)
            objective, diagnostics = score(fallback)
            return fallback, objective, {"schedule_search": "beam", **diagnostics}
        expanded.sort(key=lambda item: (item[0], item[1], item[2]))
        beam = []
        seen_partials: set[tuple[int, ...]] = set()
        for _misfit_value, _objective, _completed, partial in expanded:
            if partial in seen_partials:
                continue
            seen_partials.add(partial)
            beam.append(partial)
            if len(beam) >= beam_width:
                break

    best_fidelity: float | None = None
    best_placement: tuple | None = None
    for schedule in beam:
        spans = _spans(schedule, palette_size=palette_size, base=base, total=total)
        objective, diagnostics = scored(schedule)
        if best_fidelity is None or objective < best_fidelity:
            best_fidelity = objective
        placement_key = (_misfit(spans, target), objective, schedule)
        if best_placement is None or placement_key < best_placement[0]:
            best_placement = (placement_key, schedule, diagnostics)
    assert best_placement is not None
    return (
        best_placement[1],
        best_fidelity if best_fidelity is not None else best_placement[0][1],
        {"schedule_search": "beam", **best_placement[2]},
    )


def _minimum_spans(mass: np.ndarray, *, floor: int, printable: int) -> tuple[int, ...]:
    spans = tuple(floor if float(value) > _EMPTY_BAND_MASS else 1 for value in mass)
    if sum(spans) <= printable:
        return spans
    return (1,) * len(spans)


def _complete_schedule_from_partial(
    partial: tuple[int, ...],
    *,
    palette_size: int,
    base: int,
    total: int,
    min_spans: Sequence[int],
) -> tuple[int, ...]:
    if len(partial) >= palette_size - 1:
        return partial
    known_bounds = (base, *partial)
    spans = [
        int(known_bounds[index + 1] - known_bounds[index])
        for index in range(len(known_bounds) - 1)
    ]
    remaining_mins = list(min_spans[len(spans) :])
    remaining_layers = total - partial[-1]
    extra = max(0, remaining_layers - sum(remaining_mins))
    for index in range(extra):
        remaining_mins[index % len(remaining_mins)] += 1
    spans.extend(remaining_mins)
    return _schedule_from_spans(spans, base=base)


def _schedule_from_spans(spans: Sequence[int], *, base: int) -> tuple[int, ...]:
    schedule = []
    cursor = base
    for span in spans[:-1]:
        cursor += int(span)
        schedule.append(cursor)
    return tuple(schedule)


def _min_fidelity_descent(
    palette_size: int, base: int, total: int, score
) -> float:
    """Best achievable ΔE fidelity via multi-start coordinate descent.

    Compute the ranking score for palette sizes too large to enumerate (5-8). Placement is handled separately; this returns
    only the minimum objective found. Deterministic (fixed starts, strict-improve).
    """
    transitions = palette_size - 1
    even = even_swap_schedule(palette_size, base_layers=base, total_layers=total)
    low = tuple(base + 1 + i for i in range(transitions))  # packed toward the base
    high = tuple(total - transitions + i for i in range(transitions))  # toward top
    starts: list[tuple[int, ...]] = []
    for start in (even, low, high):
        if start not in starts:
            starts.append(start)

    best: float | None = None
    for start in starts:
        current = list(start)
        cur_obj, _ = score(tuple(current))
        for _sweep in range(_SCHEDULE_MAX_SWEEPS):
            improved = False
            for i in range(transitions):
                lo = (current[i - 1] if i > 0 else base) + 1
                hi = (current[i + 1] if i < transitions - 1 else total) - 1
                best_layer = current[i]
                local_obj = cur_obj
                for layer in range(lo, hi + 1):
                    if layer == current[i]:
                        continue
                    trial = current.copy()
                    trial[i] = layer
                    objective, _ = score(tuple(trial))
                    if objective < local_obj - 1e-12:  # strict improvement only
                        local_obj = objective
                        best_layer = layer
                if best_layer != current[i]:
                    current[i] = best_layer
                    cur_obj = local_obj
                    improved = True
            if not improved:
                break
        if best is None or cur_obj < best:
            best = cur_obj
    return best


def _placement_candidates(
    palette_size: int,
    base: int,
    total: int,
    target: Sequence[int],
) -> list[tuple[int, ...]]:
    """Deterministic candidate pool for palette sizes too large to enumerate.

    The target schedule (misfit 0) plus single-coordinate neighbors (fidelity
    tie-break room within the slack), the even schedule, and a fixed-seed sample
    of valid schedules for diversification — all judged by the corrected objective
    back in :func:`_search_placement`.
    """
    transitions = palette_size - 1
    out: set[tuple[int, ...]] = set()

    def add(schedule: Sequence[int]) -> None:
        s = tuple(int(x) for x in schedule)
        if len(s) != transitions:
            return
        if any(not (base < s[i] < total) for i in range(transitions)):
            return
        if any(s[i] >= s[i + 1] for i in range(transitions - 1)):
            return
        out.add(s)

    seed = _target_schedule(target, base)
    add(seed)
    add(even_swap_schedule(palette_size, base_layers=base, total_layers=total))
    for i in range(transitions):
        for delta in (-2, -1, 1, 2):
            neighbor = list(seed)
            neighbor[i] += delta
            add(neighbor)
    rng = random.Random(0)
    pool = list(range(base + 1, total))
    for _ in range(_SCHEDULE_SAMPLES):
        add(sorted(rng.sample(pool, transitions)))
    return list(out)


def _placement_diagnostics(
    schedule: Sequence[int], mass: np.ndarray, constants: RenderConstants
) -> dict:
    """Anti-collapse diagnostics: per-filament spans, mass, dominance, collapse flag."""
    base = constants.base_layers
    total = constants.total_layers
    spans = _spans(schedule, palette_size=len(mass), base=base, total=total)
    printable = total - base
    floor = _band_floor(printable, len(mass))
    collapsed = _collapses(spans, mass, floor=floor)
    return {
        "band_layer_spans": [int(s) for s in spans],
        "filament_mass": [round(float(m), 3) for m in mass],
        "largest_band_fraction": round(max(spans) / printable, 3) if printable else 1.0,
        "collapsed_schedule": bool(collapsed),
    }


def _diagnostics(
    bins: Sequence[_LumBin],
    bin_labs: np.ndarray,
    pred_lab: np.ndarray,
    filament_labs: np.ndarray,
    constants: RenderConstants,
) -> dict:
    target_l = bin_labs[:, 0]
    pred_l = pred_lab[:, 0]
    target_span = float(target_l.max() - target_l.min())
    pred_span = float(pred_l.max() - pred_l.min())
    coverage = pred_span / target_span if target_span > 1e-6 else 1.0

    if pred_lab.shape[0] > 1:
        adjacent = ciede2000_array(pred_lab[:-1], pred_lab[1:])
        min_contrast = float(adjacent.min())
    else:
        min_contrast = 99.0

    chroma = np.hypot(filament_labs[:, 1], filament_labs[:, 2])
    filament_l = filament_labs[:, 0]
    has_mid_neutral = bool(
        np.any((chroma < 12.0) & (filament_l > 30.0) & (filament_l < 75.0))
    )
    mid_weight = float(sum(b.weight for b in bins if 0.3 <= b.center <= 0.7))
    missing_midtone = (mid_weight > 0.15) and not has_mid_neutral

    printable_layers = constants.total_layers - constants.base_layers
    layer_starved = printable_layers < len(bins) * 0.5

    return {
        "tonal_coverage": round(coverage, 3),
        "weak_tonal_coverage": coverage < 0.7,
        "min_band_contrast": round(min_contrast, 2),
        "low_contrast": min_contrast < 5.0,
        "missing_midtone_neutral": missing_midtone,
        "layer_starved_gradient": layer_starved,
    }


def _penalties(diagnostics: dict) -> float:
    penalty = 0.0
    if diagnostics["weak_tonal_coverage"]:
        penalty += (0.7 - diagnostics["tonal_coverage"]) * 10.0
    if diagnostics["low_contrast"]:
        penalty += (5.0 - diagnostics["min_band_contrast"]) * 0.4
    if diagnostics["missing_midtone_neutral"]:
        penalty += 1.5
    if diagnostics["layer_starved_gradient"]:
        penalty += 2.0
    return penalty


# --------------------------------------------------------------------------- #
# Shared helpers (retained from the prior ranker)                             #
# --------------------------------------------------------------------------- #


def _dedupe_by_filament_name(inventory: Sequence[Filament]) -> tuple[Filament, ...]:
    seen: set[tuple[str, str]] = set()
    deduped: list[Filament] = []
    for filament in inventory:
        key = _filament_name_key(filament)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(filament)
    return tuple(deduped)


def _filament_color(filament: Filament) -> _FilamentColor:
    rgb = hex_to_rgb(filament.hex)
    return _FilamentColor(
        filament=filament,
        rgb=rgb,
        lab=rgb_to_lab(rgb),
        luminance=relative_luminance(rgb),
    )


def _fill_unique(
    selected: Sequence[Filament],
    fill_order: Sequence[Filament],
    target_size: int,
) -> tuple[Filament, ...]:
    chosen: list[Filament] = []
    seen: set[tuple[str, str, str, float]] = set()
    for filament in itertools.chain(selected, fill_order):
        key = _filament_key(filament)
        if key in seen:
            continue
        chosen.append(filament)
        seen.add(key)
        if len(chosen) == target_size:
            break
    return tuple(chosen)


def _order_filaments(
    filaments: Sequence[Filament],
    strategy: str,
    color_cache: dict,
) -> tuple[Filament, ...]:
    if strategy != "luminance_ascending":
        raise ValueError(f"unsupported order strategy: {strategy}")
    return tuple(
        sorted(
            filaments,
            key=lambda filament: (
                color_cache[_filament_key(filament)].luminance,
                filament.brand,
                filament.name,
            ),
        )
    )


def _candidate_key(filaments: Sequence[Filament]) -> tuple[tuple[str, str, str, float], ...]:
    return tuple(_filament_key(filament) for filament in filaments)


def _filament_name_key(filament: Filament) -> tuple[str, str]:
    return (normalize_key(filament.brand), normalize_key(filament.name))


def _filament_key(filament: Filament) -> tuple[str, str, str, float]:
    td = filament.td
    if not math.isfinite(td):
        td = float("inf")
    return (filament.brand, filament.name, filament.hex, td)
