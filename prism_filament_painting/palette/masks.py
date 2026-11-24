"""Apply a supplied foreground mask before palette selection."""

from PIL import Image

NEUTRAL_BG = (128, 128, 128)


def apply_foreground_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Preserve the source for an empty mask and neutralize excluded pixels otherwise."""
    grey = mask.convert("L")
    if grey.getextrema()[1] == 0:
        return image
    rgba = image.convert("RGBA")
    if grey.size != rgba.size:
        grey = grey.resize(rgba.size, Image.Resampling.NEAREST)
    alpha = grey.point(lambda v: 255 if v >= 128 else 0)
    # Keep a neutral field for consumers that discard the alpha channel.
    rgb = Image.composite(rgba.convert("RGB"), Image.new("RGB", rgba.size, NEUTRAL_BG), alpha)
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out
