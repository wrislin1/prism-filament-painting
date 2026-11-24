"""User-owned SQLite library, immutable source images and versioned project documents."""

from contextlib import closing
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4

from fastapi import HTTPException
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps, UnidentifiedImageError

from prism_filament_painting.core.forward import RenderConstants
from prism_filament_painting.palette import store
from prism_filament_painting.palette.materials import GROUPS, MIX_WARNING
from .project import canonical

MAX_UPLOAD = 16 * 1024 * 1024
MAX_PIXELS = 4_000_000


def checked_id(value):
    try:
        if str(UUID(value)) == value:
            return value
    except (ValueError, TypeError, AttributeError):
        pass
    raise HTTPException(404, "Unknown image or project.")


def read_upload(upload):
    data = upload.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File exceeds 16 MiB.")
    return data


def decoded_image(data):
    try:
        with Image.open(BytesIO(data), formats=("PNG", "JPEG", "WEBP")) as source:
            if source.width * source.height > MAX_PIXELS or getattr(source, "is_animated", False):
                raise ValueError("Choose a still image of at most four megapixels.")
            return ImageOps.exif_transpose(source).convert("RGBA")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Choose a readable PNG, JPEG or WebP image.") from exc


def write_json(path, obj):
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stage:
        temp = Path(stage.name)
        try:
            json.dump(obj, stage, allow_nan=False)
            stage.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


class LocalState:
    assets = Path(__file__).parent

    def __init__(self, directory):
        self.directory = directory
        self.projects = directory / "projects"
        self.images = directory / "images"
        self.artifacts = directory / "artifacts"
        for path in (self.projects, self.images, self.artifacts):
            path.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            store.init_db(conn)
        self.templates = Jinja2Templates(directory=self.assets / "templates")
        self.templates.env.globals.update(mixed_warning=MIX_WARNING, material_groups=GROUPS)

    def connect(self):
        return closing(store.connect_db(self.directory / "library.sqlite"))

    def add_image(self, image, name):
        key = str(uuid4())
        image.info.clear()
        image.save(self.images / (key + ".png"))
        return {"upload_token": key, "filename": Path(name or "Image").name[:180], "content_type": "image/png"}

    def image_path(self, ref):
        if not isinstance(ref, dict):
            raise ValueError("Choose a source image.")
        if ref.get("project_id"):
            ref = self.project(ref["project_id"])["image"]
        path = self.images / (checked_id(ref.get("upload_token")) + ".png")
        if not path.is_file():
            raise HTTPException(404, "Source image not found.")
        return path

    def image(self, ref):
        with Image.open(self.image_path(ref)) as image:
            return image.convert("RGBA")

    def project(self, key):
        try:
            return json.loads((self.projects / (checked_id(key) + ".json")).read_bytes())
        except FileNotFoundError:
            raise HTTPException(404, "Project not found.") from None

    def save(self, value, key=None):
        state = canonical(value)
        path = self.image_path(state["image"])
        if key is not None:
            self.project(key)
        else:
            key = str(uuid4())
        state["image"].update(project_id=key, upload_token=path.stem)
        write_json(self.projects / (key + ".json"), state)
        return {"project": self.summary(state), "state": state}

    @staticmethod
    def summary(state):
        return {"id": state["image"]["project_id"], "filename": state["image"]["filename"],
                "palette": ", ".join(s["name"] for s in state["palette"]["filaments"])}


def classic_state(ref, candidate, choice):
    defaults = RenderConstants()
    spools = []
    for spool in candidate.filaments:
        fields = asdict(spool)
        fields["td"] = fields.pop("input_distance_mm")
        spools.append(fields)
    return canonical({
        "version": 1, "image": ref,
        "palette": {"filaments": spools, "score": candidate.score, "material_group": choice.group},
        "swap_schedule": list(candidate.swap_schedule),
        "layer": {"layer_height_mm": defaults.layer_height_mm, "base_layers": defaults.base_layers,
                  "total_layers": defaults.total_layers},
        "mesh": {"max_dim_mm": 300.0, "xy_pixel_mm": 0.4, "max_triangles": 1500000,
                 "flatten_transparent": True, "orientation": "auto"},
        "luminance": {"black_point": 0., "white_point": 1., "gamma": 1.},
        "view": {"backlight_color_k": 6500, "backlight_strength": 0., "backlight_side": "back"},
    })
