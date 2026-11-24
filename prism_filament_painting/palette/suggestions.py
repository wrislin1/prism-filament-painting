"""Score spool additions and loadouts on the placements used by Prism's ranker."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from PIL import Image

from prism_filament_painting.core.forward import RenderConstants, build_forward_stack, preview_sources, render_preview
from prism_filament_painting.core.prediction import calibration_catalog
from prism_filament_painting.core.metrics import delta_e_stats

from .model import Filament
from .ranking import (
    _DEFAULT_LUM_BINS,
    _bin_target_by_luminance,
    _candidate_key,
    _dedupe_by_filament_name,
    _filament_color,
    _filament_key,
    _filament_name_key,
    _order_filaments,
    _score_fixed_palette,
)
from .scoring import build_layer_target_model

LIBRARY_CAP = 512
EXHAUSTIVE_SETS = 256
BEAM_WIDTH = 4
EXPANSION_POOL = 80
SWAP_ROUNDS = 2
MAX_SLOTS = 8
PREVIEW_EDGE = 96
MAX_RESULTS = 10


@dataclass
class Suggestion:
    filaments: tuple[Filament, ...]
    schedule: tuple[int, ...]
    score: float
    preview: Image.Image
    errors: dict[str, float]
    addition: Filament | None
    prediction_sources: list[str] | None = None
    score_sources: list[str] | None = None


@dataclass
class SearchResult:
    candidates: list[Suggestion]
    baseline: dict[str, float] | None
    possible: int
    evaluated: int
    library_size: int
    partial_evaluated: int
    baseline_sources: list[str] | None = None


def _search_sets(library, colors, slots, rank_key, limit):
    """Return the best final sets visited and distinct final/intermediate counts."""
    possible = math.comb(len(library), slots)
    indices = range(len(library))
    scores = {}

    def evaluate(key):
        if key not in scores:
            scores[key] = rank_key(tuple(library[i] for i in key))
        return scores[key]

    if slots == 1 or possible <= EXHAUSTIVE_SETS:
        winners = heapq.nsmallest(limit, combinations(indices, slots), key=evaluate)
        return winners, possible, 0

    singles = sorted(indices, key=lambda i: evaluate((i,)))
    pool = singles
    if len(pool) > EXPANSION_POOL:
        # Keep the best individual matches, then fill the expansion pool with
        # colours farthest from those already represented in Lab space.
        pool = singles[:EXPANSION_POOL // 2]
        labs = np.stack([colors[_filament_key(f)].lab for f in library])
        distances = np.full(len(library), np.inf)
        for i in pool:
            distances = np.minimum(distances, np.sum((labs - labs[i]) ** 2, axis=1))
        distances[pool] = -1
        while len(pool) < EXPANSION_POOL:
            i = int(distances.argmax())
            pool.append(i)
            distances = np.minimum(distances, np.sum((labs - labs[i]) ** 2, axis=1))
            distances[i] = -1

    beam = [(i,) for i in singles[:BEAM_WIDTH]]
    for _ in range(2, slots + 1):
        children = {tuple(sorted((*parent, i))) for parent in beam for i in pool if i not in parent}
        beam = heapq.nsmallest(BEAM_WIDTH, children, key=evaluate)

    for _ in range(SWAP_ROUNDS):
        best = beam[0]
        neighbours = {
            tuple(sorted((*best[:pos], *best[pos + 1:], i)))
            for pos in range(slots) for i in pool if i not in best
        }
        improved = min(neighbours, key=evaluate, default=best)
        if evaluate(improved) >= evaluate(best):
            break
        beam[0] = improved

    final = [key for key in scores if len(key) == slots]
    return heapq.nsmallest(limit, final, key=evaluate), len(final), len(scores) - len(final)


def suggest(
    image: Image.Image,
    inventory: Sequence[Filament],
    *,
    mode: str,
    loaded: Sequence[Filament] = (),
    slots: int = 4,
    metric: str = "mean_delta_e",
    limit: int = 5,
    constants: RenderConstants | None = None,
    search_profile: str = "alpha",
    score_profile: str = "alpha",
    beam_width: int = 8,
) -> SearchResult:
    if mode not in {"next", "loadouts", "full-set"}:
        raise ValueError("unknown suggestion mode")
    if metric not in {"mean_delta_e", "p95_delta_e"}:
        raise ValueError("metric must be mean_delta_e or p95_delta_e")
    if not 1 <= slots <= MAX_SLOTS or not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"slots must be 1–{MAX_SLOTS} and limit 1–{MAX_RESULTS}")
    library = sorted(_dedupe_by_filament_name(inventory), key=_filament_key)
    if len(library) > LIBRARY_CAP:
        raise ValueError(f"{len(library)} distinct eligible filaments exceed the {LIBRARY_CAP}-filament limit")
    if mode == "next":
        if not 1 <= len(loaded) < MAX_SLOTS or len(_dedupe_by_filament_name(loaded)) != len(loaded):
            raise ValueError(f"choose 1–{MAX_SLOTS - 1} distinct loaded filaments")
    elif loaded or slots > len(library):
        raise ValueError("choose a slot count that fits the eligible library")

    thumb = image.convert("RGBA")
    thumb.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE), Image.Resampling.BOX)
    transparent = np.asarray(thumb.getchannel("A")) == 0
    if transparent.all():
        raise ValueError("image has no visible pixels")
    target = np.asarray(thumb.convert("RGB"))
    # The rank HTTP route loads full-size RGB. Keep alpha on the preview for
    # print simulation and the visible-pixel error, as in the retained oracle.
    rank_image = image.convert("RGB")
    bins = _bin_target_by_luminance(rank_image, n_bins=_DEFAULT_LUM_BINS)
    colors = {_filament_key(f): _filament_color(f) for f in (*library, *loaded)}
    geometry = constants or RenderConstants()
    model = build_layer_target_model(rank_image, constants=geometry) if score_profile != "alpha" else None

    def measure(spools, *, publish=False, addition=None):
        ordered = _order_filaments(spools, "luminance_ascending", colors)
        candidate = _score_fixed_palette(
            rank_image, ordered, colors, geometry, bins=bins, target_model=model,
            search_profile=search_profile, score_profile=score_profile, beam_width=beam_width,
            placement_only=not publish,
        )
        stack = build_forward_stack(ordered, constants=geometry, swap_schedule=candidate.swap_schedule)
        preview = render_preview(thumb, stack)
        errors = delta_e_stats(np.asarray(preview), target, transparent_mask=transparent)
        sources = preview_sources(thumb, stack, transparent) if calibration_catalog() else None
        return Suggestion(ordered, candidate.swap_schedule, candidate.score, preview, errors, addition,
                          sources, candidate.diagnostics.get("prediction_sources"))

    baseline = None
    partial = 0
    if mode == "next":
        present = {_filament_name_key(f) for f in loaded}
        remaining = [f for f in library if _filament_name_key(f) not in present]
        baseline = measure(loaded)
        possible = evaluated = len(remaining)
        additions = (measure((*loaded, spool), addition=spool) for spool in remaining)
        best = heapq.nsmallest(limit, additions, key=lambda c: (c.errors[metric], _candidate_key(c.filaments)))
        candidates = [measure(c.filaments, publish=True, addition=c.addition) for c in best]
    else:
        possible = math.comb(len(library), slots)
        def rank_key(subset):
            item = measure(subset)
            return item.errors[metric], _candidate_key(item.filaments)

        keys, evaluated, partial = _search_sets(library, colors, slots, rank_key, limit)
        candidates = [measure([library[i] for i in key], publish=True) for key in keys]
    return SearchResult(candidates, baseline.errors if baseline else None, possible, evaluated, len(library), partial,
                        baseline.prediction_sources if baseline else None)
