"""Generate an invented three-spool inventory and a small RGB test image.

Run `python -m prism_filament_painting.palette.sample_inputs DIRECTORY` to write both examples.
The rows, including their distances, are illustrative.
"""
import argparse
from pathlib import Path

from PIL import Image


def spool_list() -> str:
    return (
        "brand,name,hex,td,material,owned\n"
        "Paper Orbit,Canal Slate,#394762,1.35,PLA,true\n"
        "Paper Orbit,Apricot Dial,#D78152,3.15,PLA,true\n"
        "Paper Orbit,Linen Kite,#E3DBBC,4.65,PLA,true\n"
    )


def sample_image() -> Image.Image:
    width, height = 31, 19
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend((255 * x // (width - 1), 255 * y // (height - 1),
                           (7 * x + 11 * y) % 256))
    return Image.frombytes("RGB", (width, height), bytes(pixels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    dest = parser.parse_args().directory
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "spools.csv").open("xb") as output:
        output.write(spool_list().encode("ascii"))
    with (dest / "ramp.png").open("xb") as output:
        sample_image().save(output, format="PNG")
