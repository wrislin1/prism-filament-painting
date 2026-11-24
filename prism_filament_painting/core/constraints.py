"""Shared editor/build constraints."""

from __future__ import annotations

LUMINANCE_BLACK_MIN = 0.0
LUMINANCE_WHITE_MAX = 1.0
LUMINANCE_GAMMA_MIN = 0.1
LUMINANCE_GAMMA_MAX = 5.0

BACKLIGHT_STRENGTH_MIN = 0.0
BACKLIGHT_STRENGTH_MAX = 1.0
BACKLIGHT_COLOR_K_MIN = 1000.0
BACKLIGHT_COLOR_K_MAX = 12000.0
BACKLIGHT_SIDE = "back"


def studio_constraints() -> dict:
    return {
        "luminance": {
            "black_point": {"min": LUMINANCE_BLACK_MIN, "max": LUMINANCE_WHITE_MAX, "default": 0.0},
            "white_point": {"min": LUMINANCE_BLACK_MIN, "max": LUMINANCE_WHITE_MAX, "default": 1.0},
            "gamma": {"min": LUMINANCE_GAMMA_MIN, "max": LUMINANCE_GAMMA_MAX, "default": 1.0},
        },
        "view": {
            "backlight_strength": {
                "min": BACKLIGHT_STRENGTH_MIN,
                "max": BACKLIGHT_STRENGTH_MAX,
                "default": 0.0,
            },
            "backlight_color_k": {
                "min": BACKLIGHT_COLOR_K_MIN,
                "max": BACKLIGHT_COLOR_K_MAX,
                "default": 6500.0,
            },
            "backlight_side": {"default": BACKLIGHT_SIDE},
        },
    }
