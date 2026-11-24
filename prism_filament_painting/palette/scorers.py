"""Interface and offline implementation for a second pass over ranked palettes."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from PIL import Image
from prism_filament_painting.core.forward import RenderConstants

from .model import PaletteCandidate


@dataclass(frozen=True)
class VisionScoreResult:
    candidates: tuple[PaletteCandidate, ...]
    suitability: dict


@runtime_checkable
class VisionScorer(Protocol):
    def score(
        self,
        image: Image.Image,
        candidates: Sequence[PaletteCandidate],
        render_constants: RenderConstants | None = None,
    ) -> VisionScoreResult:
        ...


class MockVisionScorer:
    def score(
        self,
        image: Image.Image,
        candidates: Sequence[PaletteCandidate],
        render_constants: RenderConstants | None = None,
    ) -> VisionScoreResult:
        del image, render_constants
        annotated = tuple(replace(candidate, notes="mock") for candidate in candidates)
        return VisionScoreResult(
            candidates=annotated,
            suitability={"suitable": True, "reason": "mock"},
        )
