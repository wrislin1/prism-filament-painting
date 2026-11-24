"""Use the workspace calibration directory when the maker omits the install path."""

from ..paths import default_directory
from .__main__ import main as calibrate


def main():
    calibrate(install_directory=default_directory() / "calibration")
