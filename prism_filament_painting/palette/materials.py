"""Material selection for bonded colour layers. See docs/operations/print-materials.md."""

from collections import Counter
from dataclasses import dataclass
import re
from typing import Generic, TypeVar

T = TypeVar("T")
GROUPS = ("PLA", "PETG", "PETG HT", "ASA", "ABS", "PC", "PVB", "PA", "TPU", "CPE", "HIPS", "PP")
MIX_WARNING = "Mixed materials may need different nozzle and bed temperatures. The layers may peel apart."
ALIASES = {"NYLON": "PA", "POLYCARBONATE": "PC", "RPLA": "PLA"}
UNKNOWN = "Unspecified"
UNLISTED = "Unrecognised"


def material_group(material, family=""):
    """Match polymer tokens, including finish-first labels, without guessing from colour."""
    for label in (material, family):
        tokens = re.findall(r"[A-Z0-9]+", str(label or "").upper())
        names = {ALIASES.get(token, token) for token in tokens}
        matches = names.intersection(GROUPS)
        if len(matches) > 1:
            return UNLISTED
        if matches:
            group = matches.pop()
            return "PETG HT" if group == "PETG" and "HT" in names else group
    empty_labels = {"", "unknown", "unspecified", "n/a", "none"}
    if all(str(label or "").strip().lower() in empty_labels for label in (material, family)):
        return UNKNOWN
    return UNLISTED


@dataclass(frozen=True)
class MaterialSpool(Generic[T]):
    value: T
    group: str
    owned: bool = False


@dataclass(frozen=True)
class MaterialChoice:
    group: str
    reason: str
    groups: tuple[str, ...]
    report: bool

    def values(self, spools):
        return [s.value for s in spools if self.group == "mixed" or s.group == self.group]

    def metadata(self):
        return {"material_group": self.group, "material_notice": self.reason} if self.report else {}


def choose_material(spools, requested="auto", *, loaded=()):
    """Pick within the eligible pool. A loaded next-spool palette is a hard constraint."""
    counts = Counter(s.group for s in spools)
    groups = tuple(sorted(counts.keys() | {s.group for s in loaded}))
    known = [g for g in groups if g in GROUPS]
    if requested == "mixed":
        return MaterialChoice("mixed", MIX_WARNING, groups, True)
    fixed = {s.group for s in loaded}
    if UNLISTED in fixed:
        raise ValueError("A loaded spool has an unrecognised material. Label it or choose Mix materials.")
    if len(fixed) > 1:
        raise ValueError("The loaded spools have different materials. Choose Mix materials to use them together.")
    if requested != "auto":
        if requested == UNLISTED or requested not in groups or (fixed and requested not in fixed):
            raise ValueError(f"No matching spools for material {requested} in this selection.")
        return MaterialChoice(requested, f"You chose {requested}.", groups, True)
    if fixed:
        group = next(iter(fixed))
        reason = f"Using {group} to match the loaded spools."
    elif known:
        owned = Counter(s.group for s in spools if s.owned)
        group = max(known, key=lambda g: (owned[g], counts[g]))
        owned_ties = [g for g in known if owned[g] == owned[group]]
        size_ties = [g for g in owned_ties if counts[g] == counts[group]]
        if len(known) == 1:
            reason = f"Using {group}. It is the only identified material."
        elif len(owned_ties) == 1:
            reason = f"Using {group} because it has the most owned spools ({owned[group]})."
        elif len(size_ties) == 1:
            reason = f"Owned counts tie. Using {group} because it has more eligible spools ({counts[group]})."
        else:
            reason = f"Owned and eligible spool counts tie. Using {group}, the first material alphabetically."
    else:
        group = UNKNOWN
        reason = ("No recognised material. Label the spools or choose Mix materials." if UNLISTED in groups
                  else "Material is unspecified. Check the spool labels before printing.")
    if known and (UNKNOWN in groups or UNLISTED in groups):
        reason += " Spools with missing or unrecognised material are left out."
    return MaterialChoice(group, reason, groups, len(groups) > 1 or UNLISTED in groups)
