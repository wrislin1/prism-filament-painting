"""Deterministic layer-switch variants for one fixed filament order."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from PIL import Image

from prism_filament_painting.core.forward import RenderConstants, even_swap_schedule
from prism_filament_painting.core.model import Filament

from .ranking import (
    _band_floor,
    _bin_target_by_luminance,
    _collapses,
    _DEFAULT_LUM_BINS,
    _filament_color,
    _filament_key,
    _make_score_fn,
    _misfit,
    _spans,
)


# Above this many valid schedules we sub-sample deterministically instead of
# scoring the whole exhaustive set (C(24,7) ~= 346k at 8 colors). Tunable by the
# caller via ``max_scored``; Prism resolves it from PRISM_VARIANT_MAX_SCORED.
DEFAULT_MAX_SCORED_SCHEDULES = 20000
DEFAULT_FOUR_SPOOL_LABEL = "Even four-spool schedule"
DEFAULT_FOUR_SPOOL_SCHEDULE = even_swap_schedule(4, base_layers=2, total_layers=27)


def four_spool_anchor(base_layers, total_layers):
    return even_swap_schedule(4, base_layers=base_layers, total_layers=total_layers)


@dataclass(frozen=True)
class ScoredSchedule:
    schedule: tuple[int, ...]
    spans: tuple[int, ...]
    fidelity: float
    misfit: int
    collapses: bool
    labels: tuple[str, ...] = ()


def schedule_space_size(
    palette_size: int, *, render_constants: RenderConstants | None = None
) -> int:
    """How many valid switch schedules exist for ``palette_size`` filaments.

    A schedule chooses ``palette_size - 1`` strictly-increasing switch positions
    among the printable layers, so the count is ``C(positions, transitions)``.
    Lets callers (Prism) tell whether a palette will be sampled vs enumerated
    without scoring anything.
    """
    constants = render_constants or RenderConstants()
    transitions = max(0, int(palette_size) - 1)
    positions = len(range(constants.base_layers + 1, constants.total_layers))
    if transitions == 0:
        return 1
    if positions < transitions:
        return 0
    return math.comb(positions, transitions)


def enumerate_scored_schedules(
    image: Image.Image,
    filaments: Sequence[Filament],
    *,
    render_constants: RenderConstants | None = None,
    must_include: Sequence[Sequence[int]] = (),
    max_scored: int | None = None,
) -> list[ScoredSchedule]:
    """Score valid switch schedules for one fixed filament order.

    For a palette of N filaments a schedule is a strictly-increasing choice of
    N-1 switch positions among the printable layers, so the valid set is the
    ``C(positions, N-1)`` combinations. Small palettes are enumerated
    exhaustively; once that count exceeds ``max_scored`` the space is sub-sampled
    deterministically (evenly across the full lexicographic range) so the
    selectors still see a representative spread without scoring hundreds of
    thousands of schedules. ``must_include`` schedules (anchors) are always
    scored, after being validated against the geometry — invalid ones are
    silently dropped, never raised (the candidate payload is URL-supplied).
    """
    constants = render_constants or RenderConstants()
    filament_tuple = tuple(filaments)
    palette_size = len(filament_tuple)
    if palette_size < 1:
        raise ValueError("filaments must not be empty")

    transitions = palette_size - 1
    positions = list(range(constants.base_layers + 1, constants.total_layers))
    if len(positions) < transitions:
        raise ValueError(f"not enough printable layers for palette size {palette_size}")

    cap = DEFAULT_MAX_SCORED_SCHEDULES if max_scored is None else max(1, int(max_scored))

    schedules: list[tuple[int, ...]]
    if transitions == 0:
        schedules = [()]
    else:
        total = math.comb(len(positions), transitions)
        if total <= cap:
            schedules = [tuple(combo) for combo in itertools.combinations(positions, transitions)]
        else:
            schedules = [
                _unrank_combination(index, positions, transitions)
                for index in _sample_indices(total, cap)
            ]
        # Always score validated anchors so spread_anchors finds them even when
        # the pool was sampled. Tampered/invalid anchors are dropped, not raised.
        positions_set = set(positions)
        seen = set(schedules)
        for candidate in must_include:
            anchor = _valid_anchor(candidate, positions_set, transitions)
            if anchor is not None and anchor not in seen:
                schedules.append(anchor)
                seen.add(anchor)

    color_cache = {_filament_key(f): _filament_color(f) for f in filament_tuple}
    bins = _bin_target_by_luminance(image, n_bins=_DEFAULT_LUM_BINS)
    score, mass, target = _make_score_fn(filament_tuple, color_cache, bins, constants)
    floor = _band_floor(constants.total_layers - constants.base_layers, palette_size)

    out: list[ScoredSchedule] = []
    for schedule in schedules:
        spans = _spans(
            schedule,
            palette_size=palette_size,
            base=constants.base_layers,
            total=constants.total_layers,
        )
        fidelity, _ = score(schedule)
        out.append(
            ScoredSchedule(
                schedule=tuple(int(v) for v in schedule),
                spans=spans,
                fidelity=fidelity,
                misfit=_misfit(spans, target),
                collapses=_collapses(spans, mass, floor=floor),
            )
        )
    return out


def _sample_indices(total: int, cap: int) -> list[int]:
    """Deterministic, evenly-spaced indices across the full ``[0, total-1]`` range.

    Spans BOTH endpoints (index 0 and ``total-1``) so the sample is not biased
    away from the last combination — ``(i * total) // cap`` would never reach it.
    De-duped + sorted for stability (the closed form can repeat near the ends).
    """
    if cap >= total:
        return list(range(total))
    if cap <= 1:
        return [0]
    return sorted({(i * (total - 1)) // (cap - 1) for i in range(cap)})


def _unrank_combination(index: int, positions: Sequence[int], k: int) -> tuple[int, ...]:
    """The ``index``-th k-combination of ``positions`` in lexicographic order.

    Matches ``itertools.combinations(positions, k)`` ordering via the
    combinatorial number system, so index 0 -> first combo, total-1 -> last.
    """
    n = len(positions)
    result: list[int] = []
    remaining = index
    start = 0
    for chosen in range(k):
        for j in range(start, n):
            block = math.comb(n - j - 1, k - chosen - 1)
            if remaining < block:
                result.append(positions[j])
                start = j + 1
                break
            remaining -= block
    return tuple(result)


def _valid_anchor(
    schedule: Sequence[int], positions_set: set[int], transitions: int
) -> tuple[int, ...] | None:
    """Coerce an externally-supplied schedule into a valid anchor, or None.

    Guards tampered URL candidate payloads: wrong length, out-of-range positions,
    or non-strictly-increasing schedules are dropped (return None), never raised.
    """
    try:
        values = [int(v) for v in schedule]
    except (TypeError, ValueError):
        return None
    if len(values) != transitions:
        return None
    if any(v not in positions_set for v in values):
        return None
    if any(later <= earlier for earlier, later in zip(values, values[1:])):
        return None
    return tuple(values)


def select_top_k(scored: Sequence[ScoredSchedule], k: int) -> list[ScoredSchedule]:
    pool = _selection_pool(scored)
    return _clamp(pool, k, key=_quality_key)


def select_around_optimal(scored: Sequence[ScoredSchedule], k: int) -> list[ScoredSchedule]:
    pool = _selection_pool(scored)
    if not pool:
        _validate_k(k)
        return []
    optimal = min(pool, key=_quality_key)
    rest = sorted(
        (s for s in pool if s.schedule != optimal.schedule),
        key=lambda s: (_schedule_distance(s.schedule, optimal.schedule), *_quality_key(s)),
    )
    return _limit([optimal, *rest], k)


def select_spread(scored: Sequence[ScoredSchedule], k: int) -> list[ScoredSchedule]:
    pool = _selection_pool(scored)
    _validate_k(k)
    if not pool:
        return []
    selected = [min(pool, key=_quality_key)]
    while len(selected) < min(k, len(pool)):
        selected_schedules = {s.schedule for s in selected}
        remaining = [s for s in pool if s.schedule not in selected_schedules]
        selected.append(
            max(
                remaining,
                key=lambda s: (
                    min(_schedule_distance(s.schedule, kept.schedule) for kept in selected),
                    -s.fidelity,
                    -s.misfit,
                    tuple(-v for v in s.schedule),
                ),
            )
        )
    return selected


def select_spread_with_anchors(
    scored: Sequence[ScoredSchedule],
    k: int,
    *,
    constants: RenderConstants,
    ranked_schedule: Sequence[int] | None = None,
    four_spool_schedule=four_spool_anchor,
    four_spool_label=DEFAULT_FOUR_SPOOL_LABEL,
) -> list[ScoredSchedule]:
    _validate_k(k)
    by_schedule = {s.schedule: s for s in scored}
    labels: dict[tuple[int, ...], list[str]] = {}

    def add(schedule: Sequence[int], label: str) -> None:
        s = tuple(int(v) for v in schedule)
        if s in by_schedule:
            labels.setdefault(s, []).append(label)

    if scored:
        add(min(scored, key=_quality_key).schedule, "Best fidelity")
        add(min(scored, key=lambda s: (s.misfit, s.fidelity, s.schedule)).schedule, "Mass target")
    if ranked_schedule:
        add(ranked_schedule, "Ranked recommendation")

    palette_size = len(scored[0].spans) if scored else 0
    if palette_size:
        add(
            even_swap_schedule(
                palette_size,
                base_layers=constants.base_layers,
                total_layers=constants.total_layers,
            ),
            "Even",
        )
    if palette_size == 4:
        default_4_anchor = four_spool_schedule(
            constants.base_layers, constants.total_layers
        )
        if default_4_anchor is not None:
            add(default_4_anchor, four_spool_label)

    anchors = [
        by_schedule[schedule]
        for schedule in labels
        if schedule in by_schedule
    ]
    selected = anchors[: min(len(anchors), k)]
    if len(selected) < min(k, len(scored)):
        selected_schedules = {s.schedule for s in selected}
        for candidate in select_spread(scored, k):
            if candidate.schedule not in selected_schedules:
                selected.append(candidate)
                selected_schedules.add(candidate.schedule)
            if len(selected) >= min(k, len(scored)):
                break
    return _with_labels(selected, labels)


def select_variants(
    strategy: str,
    scored: Sequence[ScoredSchedule],
    k: int,
    *,
    constants: RenderConstants,
    ranked_schedule: Sequence[int] | None = None,
    four_spool_schedule=four_spool_anchor,
    four_spool_label=DEFAULT_FOUR_SPOOL_LABEL,
) -> list[ScoredSchedule]:
    if strategy == "spread_anchors":
        return select_spread_with_anchors(
            scored, k, constants=constants, ranked_schedule=ranked_schedule,
            four_spool_schedule=four_spool_schedule, four_spool_label=four_spool_label,
        )
    try:
        selector = STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown variant strategy: {strategy}") from exc
    return selector(scored, k)


def _selection_pool(scored: Sequence[ScoredSchedule]) -> list[ScoredSchedule]:
    pool = [s for s in scored if not s.collapses]
    return pool or list(scored)


def _quality_key(schedule: ScoredSchedule):
    return (schedule.fidelity, schedule.misfit, schedule.schedule)


def _schedule_distance(a: Sequence[int], b: Sequence[int]) -> int:
    return int(sum(abs(int(x) - int(y)) for x, y in zip(a, b, strict=True)))


def _validate_k(k: int) -> None:
    if int(k) < 1:
        raise ValueError("k must be at least 1")


def _limit(scored: Sequence[ScoredSchedule], k: int) -> list[ScoredSchedule]:
    _validate_k(k)
    return list(scored[: min(int(k), len(scored))])


def _clamp(
    scored: Sequence[ScoredSchedule],
    k: int,
    *,
    key: Callable[[ScoredSchedule], object],
) -> list[ScoredSchedule]:
    _validate_k(k)
    return sorted(scored, key=key)[: min(int(k), len(scored))]


def _with_labels(
    selected: Sequence[ScoredSchedule],
    labels: dict[tuple[int, ...], list[str]],
) -> list[ScoredSchedule]:
    return [
        replace(schedule, labels=tuple(labels.get(schedule.schedule, ())))
        for schedule in selected
    ]


STRATEGIES: dict[str, Callable[[Sequence[ScoredSchedule], int], list[ScoredSchedule]]] = {
    "spread": select_spread,
    "top_k": select_top_k,
    "around_optimal": select_around_optimal,
}
