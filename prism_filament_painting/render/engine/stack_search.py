"""Bounded order and swap choices for opt-in stack optimization."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import replace
from itertools import combinations, permutations
from math import comb
from random import Random

from prism_filament_painting.core.model import Filament

from prism_filament_painting.render.errors import QuiltInputError

from .params import LayerParams, OptimizeParams
from .swaps import derive_swap_schedule


MAX_ORDERS = 24
MAX_SCHEDULES = 64


def limited_layers(layers: LayerParams, params: OptimizeParams) -> LayerParams:
    cap = params.max_layers
    if cap is None:
        return layers
    if type(cap) is not int or cap <= layers.base_layers:
        raise QuiltInputError("max_layers must be an integer greater than base_layers")
    return replace(layers, total_layers=min(layers.total_layers, cap))


def _priorities(
    indices: tuple[int, ...], filaments: Sequence[Filament], pinned: set[Filament], seed: int,
) -> Iterator[tuple[int, ...]]:
    yield indices
    # If the distinct-spool budget binds, filling from pinned spools and then
    # the most frequent spools realizes the longest possible legal order.
    counts = Counter(filaments[i] for i in indices)
    yield tuple(sorted(indices, key=lambda i: (filaments[i] not in pinned, -counts[filaments[i]])))
    if len(indices) <= 4:
        yield from permutations(indices)
    else:
        rng = Random(seed)
        for _ in range(MAX_ORDERS):
            yield tuple(rng.sample(indices, len(indices)))


def stack_orders(
    filaments: Sequence[Filament], layers: LayerParams, params: OptimizeParams,
    explicit: Sequence[int] | None,
) -> Iterator[tuple[int, ...]]:
    """Yield legal subsets of the supplied bottom-to-top palette entries."""
    n = len(filaments)
    if params.max_swaps is not None and (type(params.max_swaps) is not int or params.max_swaps < 0):
        raise QuiltInputError("max_swaps must be a non-negative integer")
    if params.max_filaments is not None and (type(params.max_filaments) is not int or params.max_filaments < 1):
        raise QuiltInputError("max_filaments must be a positive integer")
    distinct_limit = params.max_filaments or n
    largest = min(n, layers.total_layers - layers.base_layers)
    if params.max_swaps is not None:
        largest = min(largest, params.max_swaps + 1)

    if explicit is not None:
        derive_swap_schedule(len(explicit) + 1, total_layers=layers.total_layers,
                             base_layers=layers.base_layers, explicit=explicit)
    pins = {pos: i for pos, i in enumerate(params.pinned_filaments) if i is not None}
    if any(type(i) is not int or not 0 <= i < n for i in pins.values()):
        raise QuiltInputError("pinned_filaments must contain null or a zero-based palette index")
    reserved = set(pins.values())
    if len(reserved) != len(pins):
        raise QuiltInputError("a palette entry cannot be pinned to two positions")
    pinned_spools = {filaments[i] for i in reserved}
    smallest = max(pins, default=0) + 1
    if len(pinned_spools) > distinct_limit or smallest > largest:
        raise QuiltInputError("pinned positions cannot fit the filament, swap or layer limits")
    if params.pin_order:
        if (
            n > largest or len(set(filaments)) > distinct_limit
            or any(pos != i for pos, i in pins.items())
            or (explicit is not None and len(explicit) != n - 1)
        ):
            raise QuiltInputError("pin_order conflicts with stack limits, pinned positions or swap_schedule")
        yield tuple(range(n))
        return
    if explicit is not None:
        size = len(explicit) + 1
        if not smallest <= size <= largest:
            raise QuiltInputError("explicit swap_schedule cannot fit the pinned positions or stack limits")
        smallest = largest = size
    elif params.max_filaments is None and params.max_swaps is None and params.max_layers is None:
        # A wider search or a position pin alone preserves the submitted stack size.
        if n > largest:
            raise QuiltInputError("not enough printable layers for the submitted stack")
        smallest = largest = n

    indices = tuple(i for i in range(n) if i not in reserved)
    seen: set[tuple[int, ...]] = set()
    for priority in _priorities(indices, filaments, pinned_spools, params.seed):
        selected = []
        spools = pinned_spools.copy()
        for i in priority:
            if len(spools) < distinct_limit or filaments[i] in spools:
                selected.append(i)
                spools.add(filaments[i])
        longest = min(largest, len(selected) + len(pins))
        if longest < smallest:
            continue
        # Spread the order budget across different permutations even when the
        # palette admits many lengths. Include both ends of the legal range.
        sizes = range(longest, smallest - 1, -1)
        if len(sizes) > 4:
            sizes = (longest, longest - 1, (longest + smallest) // 2, smallest)
        for size in sizes:
            free = iter(selected)
            order = tuple(pins[pos] if pos in pins else next(free) for pos in range(size))
            if order not in seen:
                seen.add(order)
                yield order
                if len(seen) == MAX_ORDERS:
                    return
    if not seen:
        raise QuiltInputError("no stack fits the distinct-filament limit and required positions or swaps")


def baseline_schedule(
    size: int, layers: LayerParams, explicit: Sequence[int] | None, seed: Sequence[int] | None,
) -> tuple[int, ...]:
    """An explicit schedule is binding. A seed only applies if it fits this stack."""
    if explicit is None and seed is not None:
        try:
            return derive_swap_schedule(size, total_layers=layers.total_layers,
                                        base_layers=layers.base_layers, explicit=seed)
        except QuiltInputError:
            pass
    return derive_swap_schedule(size, total_layers=layers.total_layers,
                                base_layers=layers.base_layers, explicit=explicit)


def wide_schedules(
    size: int, layers: LayerParams, explicit: Sequence[int] | None, seed: Sequence[int] | None, limit: int,
) -> list[tuple[int, ...]]:
    if explicit is not None:
        return [tuple(explicit)]
    even = derive_swap_schedule(size, total_layers=layers.total_layers, base_layers=layers.base_layers)
    schedules = {even: None}
    limit = min(MAX_SCHEDULES, max(1, limit))
    if seed is not None:
        valid = baseline_schedule(size, layers, None, seed)
        if limit == 1:
            return [valid]
        schedules[valid] = None
    swaps = size - 1
    legal = range(layers.base_layers + 1, layers.total_layers)
    if comb(len(legal), swaps) <= limit:
        schedules.update(dict.fromkeys(combinations(legal, swaps)))
    else:
        # Move boundaries jointly, including early/late clusters. Randomness
        # is local and fixed so search is reproducible across requests.
        rng = Random(0)
        for _ in range(2 * limit):
            if len(schedules) >= limit:
                break
            schedules[tuple(sorted(rng.sample(legal, swaps)))] = None
    return list(schedules)
