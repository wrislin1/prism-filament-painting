import pytest

from prism_filament_painting.core.color import ciede2000, hex_to_lab, rgb_to_lab
from prism_filament_painting.core.forward import surface_colors_by_layer
from prism_filament_painting.core.model import Filament
from prism_filament_painting.core.opacity import measure_crossing, threshold_divisor


@pytest.mark.parametrize(
    "top,backing,td,expected_a,scale,expected_patch",
    [
        ("#FFFFFF", "#000000", 5.7, 1.40909, 0.1, 11),
        ("#000000", "#FFFFFF", 3.2, 1.31341, 0.1, 6),
        ("#F9487C", "#000000", 4.1, 1.41767, 0.1, 8),
        ("#FFFFFF", "#000000", 5.7, 1.40909, 0.17, 18),
        ("#000000", "#FFFFFF", 3.2, 1.31341, 0.17, 9),
        ("#F9487C", "#000000", 4.1, 1.41767, 0.17, 13),
    ],
)
def test_forward_round_trip(top, backing, td, expected_a, scale, expected_patch):
    colors = surface_colors_by_layer(
        [Filament("Test", "Base", backing, 1), Filament("Test", "Top", top, td)],
        (1,),
        layer_height_mm=0.08,
        total_layers=30,
        front_lit_scale=scale,
    )
    differences = [ciede2000(rgb_to_lab(rgb), hex_to_lab(top)) for rgb in colors]
    first = next(i for i, difference in enumerate(differences) if difference <= 2)
    assert first == expected_patch and max(differences[first:]) <= 2
    a = threshold_divisor(top, backing)
    assert a == pytest.approx(expected_a, abs=0.00001)
    result = measure_crossing(a, [first] * 3, [td] * 5)
    assert result["scale"][0] < scale <= result["scale"][1]
    assert result["hiding_distance_interval_mm"][0] < td * scale <= result["hiding_distance_interval_mm"][1]
    assert result["crossing_mm"] == pytest.approx(((first - 1) * 0.08, first * 0.08))


def test_meter_spread_accuracy_and_pass_envelope():
    result = measure_crossing(2, [10, 12, 11], [4, 4.4, 4.2, 4.1, 4.5], {9: 0.71, 12: 0.97})
    assert result["median_td_mm"] == 4.2
    assert result["crossing_mm"] == (0.71, 0.97)
    assert result["scale_with_accuracy"] == pytest.approx(
        (0.71 / 2 / (4.5 / 0.925), 0.97 / 2 / (4 / 1.075))
    )


@pytest.mark.parametrize(
    "passes,expected",
    [
        ([1, 1, 1], (0, 0.08)),
        ([None] * 3, (2.4, None)),
        ([30, None, 28], (27 * 0.08, None)),
        ([1, 28, None], (0, None)),
    ],
)
def test_censored_readings(passes, expected):
    result = measure_crossing(2, passes)
    assert result["crossing_mm"] == expected
    assert result["scale"] is None and result["scale_with_accuracy"] is None


def test_reject_unsuitable_backing_and_invalid_inputs():
    with pytest.raises(ValueError, match="backing"):
        threshold_divisor("101010", "111111")
    for passes in ([1, 2], [True, 2, 3], [0, 2, 3], [2, 31, 3]):
        with pytest.raises(ValueError):
            measure_crossing(2, passes)
    for scans in ([1], [float("inf")] * 5, [0] * 5):
        with pytest.raises(ValueError):
            measure_crossing(2, [2] * 3, scans)
    with pytest.raises(ValueError, match="increase"):
        measure_crossing(2, [2] * 3, measured_heights={1: 0.3})
