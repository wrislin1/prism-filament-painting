"""Resolve palette colour targets against active, available filament records."""
from dataclasses import asdict

from prism_filament_painting.palette.store import material_spools, SOURCE_PERSONAL, filaments_for_ranking, list_filaments
from prism_filament_painting.palette.materials import choose_material
from prism_filament_painting.core.color import ciede2000_array, hex_to_rgb, rgb_array_to_lab

_TARGETS = {
    "Black, grey and white": (("black", "#181818"), ("grey", "#888888"), ("white", "#f5f5f5")),
    "Warm earth": (("brown", "#75452e"), ("ochre", "#c58940"), ("cream", "#f4e7c6")),
    "Cool blue": (("navy", "#223651"), ("blue", "#487ba8"), ("ice", "#d8e9f2")),
}



def _wire_filament(spool):
    return {("td" if key == "input_distance_mm" else key): value for key, value in asdict(spool).items()}


def resolved_presets(conn, requested="auto"):
    spools = material_spools(list_filaments(conn, available_only=True))
    choice = choose_material(spools, requested)
    rows = choice.values(spools)
    candidates = []
    for owned in (True, False):
        pool = [row for row in rows if (row.source == SOURCE_PERSONAL) == owned]
        for filament in filaments_for_ranking(pool, conn=conn, provenance=True):
            candidates.append((owned, filament))
    labs = rgb_array_to_lab([hex_to_rgb(f.hex) for _, f in candidates]) if candidates else None
    presets = []
    for name, targets in _TARGETS.items():
        used = set()
        filaments, missing = [], []
        for label, color in targets:
            target = rgb_array_to_lab([hex_to_rgb(color)])[0]
            distances = ciede2000_array(target, labs) if labs is not None else []
            choices = []
            for index, delta in enumerate(distances):
                owned, filament = candidates[index]
                if delta <= 15 and filament.hex.lower() not in used:
                    choices.append((not owned, float(delta), index))
            if choices:
                _, _, index = min(choices)
                filament = candidates[index][1]
                used.add(filament.hex.lower())
                filaments.append(_wire_filament(filament))
            else:
                missing.append(label)
        presets.append({"name": name, "filaments": filaments, "missing": missing})
    return {"presets": presets, **choice.metadata()}
