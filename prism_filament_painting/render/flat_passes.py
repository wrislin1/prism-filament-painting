"""Plan one pass per occupied slot in each deposited layer."""

from itertools import permutations
from math import isfinite


def count_changes(passes, *, start):
    count = 0
    previous = start
    for layer in passes:
        for slot in layer:
            count += previous != slot
            previous = slot
    return count


def plan_passes(usage, *, start, costs=None):
    """Minimize directed cost, then changes, then slot sequence; include layer boundaries."""
    states = {start: (0, 0, ())}
    for layer in usage:
        following = {}
        for order in permutations(layer):
            for previous, (cost, changes, path) in states.items():
                edges = tuple(zip((previous, *order[:-1]), order))
                switches = sum(a != b for a, b in edges)
                extra = switches if costs is None else sum(costs[a-1][b-1] for a, b in edges)
                choice = (cost + extra, changes + switches, path + (order,))
                end = order[-1]
                if end not in following or choice < following[end]:
                    following[end] = choice
        states = following
    cost, _, passes = min(states.values())
    return passes, cost


def validate_costs(costs, n):
    if costs is None:
        return
    if len(costs) != n or any(len(row) != n for row in costs):
        raise ValueError("supply one transition cost per ordered pair of loaded slots")
    for i, row in enumerate(costs):
        for j, value in enumerate(row):
            if not isfinite(value) or value < 0 or (i == j and value != 0):
                raise ValueError("transition costs must be finite, non-negative and zero on the diagonal")
