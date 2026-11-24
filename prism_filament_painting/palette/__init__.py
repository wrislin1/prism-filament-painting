"""Deterministic filament selection and printability analysis."""
from .analyze import PrintabilityReport, analyze_printability, resolved_bands
from .enhance import EnhanceParams, apply_enhancement, recommend_enhancement
from .inventory import parse_inventory
from .model import Filament, PaletteCandidate
from .ranking import rank_palettes, score_palette
