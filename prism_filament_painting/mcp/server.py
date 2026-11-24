"""Read-only Prism palette tools for local stdio clients."""

from __future__ import annotations

import csv
import io
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP, Image as MCPImage
from mcp.types import ImageContent, TextContent, ToolAnnotations
from prism_filament_painting.palette import Filament, rank_palettes
from prism_filament_painting.palette.inventory import iter_inventory_dicts, spool_from_row
from prism_filament_painting.palette.materials import MaterialSpool, choose_material, MIX_WARNING
from prism_filament_painting.core import build_forward_stack, render_preview
from PIL import Image, ImageOps
from pydantic import Field, StrictInt

MAX_FILE = 8 * 1024 * 1024
MAX_EDGE = 4096
MAX_PIXELS = 12_000_000
THUMB_EDGE = 224
MAX_PNG = 48 * 1024
MAX_SPOOLS = 128
MAX_PALETTE = 4
PRIVACY = (
    "Before using an image tool, tell the user that source thumbnails and simulated "
    "previews are visible to their assistant and can be sent to its provider. "
    "Use only images they agree to share."
)
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
ImagePath = Annotated[str, Field(min_length=1, max_length=4096)]
SpoolId = Annotated[StrictInt, Field(ge=1, le=MAX_SPOOLS)]



def _wire_filament(spool):
    return {("td" if key == "input_distance_mm" else key): value for key, value in asdict(spool).items()}


def regular_file_bytes(path: Path, limit: int) -> bytes:
    """Bound the actual read even if a file is replaced or grows during access."""
    try:
        path = path.expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("Expected a regular file.")
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Expected a regular file.")
            if info.st_size <= limit:
                data = handle.read(limit + 1)
                if len(data) <= limit:
                    return data
    except FileNotFoundError:
        raise ValueError("File does not exist.") from None
    except (OSError, RuntimeError):
        raise ValueError("Cannot resolve or read the local file.") from None
    raise ValueError(f"File exceeds the {limit}-byte limit.")


def load_image(path: str) -> Image.Image:
    raw = regular_file_bytes(Path(path), MAX_FILE)
    try:
        with Image.open(io.BytesIO(raw), formats=("PNG", "JPEG", "WEBP")) as image:
            if max(image.size) > MAX_EDGE or image.width * image.height > MAX_PIXELS:
                raise ValueError("Image exceeds 4096 pixels per side or 12 megapixels.")
            if getattr(image, "is_animated", False):
                raise ValueError("Choose a single still frame.")
            image = ImageOps.exif_transpose(image).convert("RGBA")
            image.thumbnail((THUMB_EDGE, THUMB_EDGE), Image.Resampling.LANCZOS)
            image.info.clear()
            return image
    except Image.DecompressionBombError:
        raise ValueError("Image exceeds the pixel limit.") from None
    except (OSError, SyntaxError):
        raise ValueError("Expected a readable PNG, JPEG or WebP image.") from None


def png_content(image: Image.Image) -> ImageContent:
    # Detailed source thumbnails can exceed the byte budget even at 224 pixels.
    while True:
        out = io.BytesIO()
        image.save(out, format="PNG")
        if out.tell() <= MAX_PNG:
            return MCPImage(data=out.getvalue(), format="png").to_image_content()
        image = image.resize(tuple(max(1, n * 3 // 4) for n in image.size), Image.Resampling.LANCZOS)


@dataclass(frozen=True)
class Spool:
    id: int
    filament: Filament
    owned: bool
    group: str

    def metadata(self) -> dict[str, object]:
        return {"id": self.id, **_wire_filament(self.filament), "owned": self.owned}


def load_library(path: Path | None) -> list[Spool]:
    if path is None:
        return []
    raw = regular_file_bytes(path, 512 * 1024).decode("utf-8-sig")
    if path.suffix.lower() == ".json":
        rows = iter_inventory_dicts(json.loads(raw))
    elif path.suffix.lower() == ".csv":
        rows = csv.DictReader(io.StringIO(raw, newline=""))
    else:
        raise ValueError("Library must be a CSV or JSON inventory.")
    spools = []
    for index, row in enumerate(rows, 1):
        if index > MAX_SPOOLS:
            raise ValueError(f"Choose a library of at most {MAX_SPOOLS} spools.")
        entry = spool_from_row(row, row_number=index)
        filament = entry.value
        if max(len(filament.brand), len(filament.name)) > 64:
            raise ValueError(f"Row {index}: limit brand and name to 64 characters each.")
        spools.append(Spool(index, filament, entry.owned, entry.group))
    return spools


def predicted(
    source: Image.Image,
    filaments: list[Filament] | tuple[Filament, ...],
    swaps: list[int] | tuple[int, ...],
) -> ImageContent:
    stack = build_forward_stack(filaments, swap_schedule=swaps)
    return png_content(render_preview(source, stack))


def create_server(library: Path | None = None) -> FastMCP:
    spools = load_library(library)
    server = FastMCP("prism-palette", instructions=PRIVACY)

    @server.tool(annotations=READ_ONLY)
    def list_filaments(
        owned_only: bool = False,
        offset: Annotated[int, Field(ge=0, le=MAX_SPOOLS)] = 0,
        limit: Annotated[int, Field(ge=1, le=16)] = 16,
    ) -> dict[str, object]:
        """Page through the configured library. IDs identify spools for rendering."""
        selected = [s for s in spools if not owned_only or s.owned]
        page = selected[offset : offset + limit]
        choice = choose_material([MaterialSpool(s, s.group, s.owned) for s in selected])
        result = {"total": len(selected), "filaments": [s.metadata() for s in page]}
        if choice.report:
            result["materials"] = list(choice.groups)
            for row, spool in zip(result["filaments"], page):
                row["material"] = spool.group
        return result

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def rank_candidate_palettes(
        image_path: ImagePath,
        palette_size: Annotated[int, Field(ge=1, le=MAX_PALETTE)] = 2,
        candidates: Annotated[int, Field(ge=1, le=3)] = 3,
        material_group: str = "auto",
        owned_only: bool = False,
    ) -> list[TextContent | ImageContent]:
        """Share a source thumbnail, then each ranked stack's metadata and simulated PNG with the assistant."""
        pool = [s for s in spools if not owned_only or s.owned]
        if not spools:
            raise ValueError("Library is empty. Supply a spool list with --library.")
        if not pool:
            raise ValueError("No spools are marked owned in the configured library.")
        entries = [MaterialSpool(s, s.group, s.owned) for s in pool]
        choice = choose_material(entries, material_group)
        pool = choice.values(entries)
        source = load_image(image_path)
        by_filament = {s.filament: s for s in reversed(pool)}
        ranked = rank_palettes(
            source, [s.filament for s in pool], palette_size=palette_size, max_candidates=candidates,
        )
        result = [TextContent(type="text", text="Source thumbnail" + (" · " + choice.reason if choice.report else "")), png_content(source)]
        for number, candidate in enumerate(ranked, 1):
            metadata = {
                "candidate": number,
                "filaments": [by_filament[f].metadata() for f in candidate.filaments],
                "swap_layers": candidate.swap_schedule,
                "score": candidate.score,
            }
            result.append(TextContent(type="text", text=json.dumps(metadata, ensure_ascii=False)))
            result.append(predicted(source, candidate.filaments, candidate.swap_schedule))
        return result

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def render_palette_preview(
        image_path: ImagePath,
        filament_ids: Annotated[list[SpoolId], Field(min_length=1, max_length=MAX_PALETTE)],
        swap_layers: Annotated[list[StrictInt], Field(max_length=MAX_PALETTE - 1)],
        mix_materials: bool = False,
    ) -> list[TextContent | ImageContent]:
        """Share a simulated PNG for bottom-to-top spool IDs and increasing swap layers (3–26)."""
        if any(i > len(spools) for i in filament_ids):
            raise ValueError("Unknown filament ID. Use list_filaments for this library.")
        entries = [MaterialSpool(spools[i - 1].filament, spools[i - 1].group) for i in filament_ids]
        choice = choose_material(entries, "mixed" if mix_materials else "auto", loaded=entries)
        preview = predicted(load_image(image_path), choice.values(entries), swap_layers)
        return [TextContent(type="text", text=MIX_WARNING), preview] if mix_materials else [preview]

    return server
