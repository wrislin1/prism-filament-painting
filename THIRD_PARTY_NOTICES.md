# Prism licences and credits

Prism's original code, tests, documentation, examples and workspace stylesheet are copyright © 2026 wrislin1, under the [MIT licence](LICENSE). The stylesheet uses system fonts. The package build includes the hand-written CSS directly.

## Layer events

The layer-event XML and operation numbers in `prism_filament_painting/render/engine/classic_project.py` include an MIT adaptation of `src/layertone/threemf.py`, copyright (c) 2026 Akoozie. The module retains the upstream credit. [Layertone at 836090ede4c0679cd05e56b49793f6562e9200a0](https://github.com/Akoozie/layertone/blob/836090ede4c0679cd05e56b49793f6562e9200a0/src/layertone/threemf.py) · [Full MIT text](licenses/layertone-MIT.txt)

## User-imported catalogue data

The original reader in `prism_filament_painting/palette/catalogues.py` accepts user-supplied data. These notices cover imported records from the named revisions. The distribution contains zero catalogue records.

| Data origin and revision | Copyright | Full licence |
| --- | --- | --- |
| [OpenPrintTag material database, 07787de895eab8d223b480e567c1d058397e59f8](https://github.com/OpenPrintTag/openprinttag-database/tree/07787de895eab8d223b480e567c1d058397e59f8) | Copyright 2025 PRUSA RESEARCH A.S. | [MIT](licenses/catalogues/openprinttag-database-MIT.txt) |
| [Open Filament Database, d1b75e8b7739a8991e0e3bfb600acc161c340776](https://github.com/OpenFilamentCollective/open-filament-database/tree/d1b75e8b7739a8991e0e3bfb600acc161c340776) | Copyright © 2025 OpenFilamentCollective | [MIT](licenses/catalogues/open-filament-database-MIT.txt) |

## Published equations

The original wedge solver in `prism_filament_painting/core/opacity.py` uses the base-10 attenuation law and its inversion at a two-unit colour difference documented in Kromacut's calibration theory. The forward model in `core/forward.py` starts with a front-lit hiding distance of one tenth of published transmission distance. This is an approximation. The published account describes a nonlinear conversion. [Calibration theory](https://kromacut.com/docs/calibration-theory) · [Front-lit observations, July 2023](https://threadreaderapp.com/thread/1682029289875865606.html)

## Direct dependencies

The installer obtains these distributions separately under their publishers' licences. The build requirement applies when building from source. The platform conditions select the CPU build of PyTorch on Linux and Windows.

| Scope | Declared requirement | Upstream licence |
| --- | --- | --- |
| base | fastapi==0.136.3 | MIT |
| base | uvicorn==0.48.0 | BSD-3-Clause |
| base | jinja2==3.1.6 | BSD-3-Clause |
| base | python-multipart==0.0.30 | Apache-2.0 |
| base | pydantic>=2.11,<3 | MIT |
| base | numpy==2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| base | pillow==12.2.0 | MIT-CMU |
| base | colour-science==0.4.7 | BSD-3-Clause |
| base | trimesh==4.12.2 | MIT |
| optimize | torch==2.9.1+cpu; sys_platform == 'linux' or sys_platform == 'win32' | BSD-3-Clause |
| optimize | torch==2.9.1; sys_platform != 'linux' and sys_platform != 'win32' | BSD-3-Clause |
| catalogues | PyYAML==6.0.3 | MIT |
| mcp | mcp==1.30.0 | MIT |
| build | setuptools>=77 | MIT |

The Linux NumPy wheel inspected includes libgfortran under the GNU General Public License with the GCC Runtime Library Exception and libquadmath under the GNU Lesser General Public License. Pillow's wheel licence and the CPU PyTorch wheel's LICENSE and NOTICE give additional terms for bundled components. A distributor shipping an installer, container or wheelhouse needs a separate inventory of the native libraries, codecs and notices in that bundle.
