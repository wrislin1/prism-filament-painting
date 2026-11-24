# Prism filament painting

Design layered reliefs and flat multi-material prints with a browser, Python 3.12 and a CPU. Install the wheel in a virtual environment and run `prism-filament-painting`. Open the loopback address printed in the terminal. The library starts empty. Use `--data-dir DIRECTORY` to choose where Prism keeps your spools, source images, projects, calibrations and print files.

1. Open **Find a palette**. Choose **Add your spools** to enter the brand, colour, material and transmission distance for a spool you have. Or choose **Load invented sample** to add three example spools and select the practice picture.
2. Choose **Find matching palettes** with the sample picture selected, or choose your own picture. Open a palette to see its predicted print. Use **Save project** to keep the picture and layer settings. You can stop the launcher, start it again with the same data directory, and reopen the print from **Saved projects**.
3. Choose **Export print files** in the editor. Open the STL in your slicer, use the change list for the layer numbers and heights, and view the PNG to check the predicted colours.
4. For a flat print, open **Flat print**, choose a picture and match the four slots to the spools in your printer. **Download sample picture** on the palette page supplies a picture you can choose here too. Generate a preview, adjust a region’s colours if needed and generate it again. Download the 3MF geometry and slot map and apply the settings in your slicer.

The sample spools and picture can also be written to disk with `python -m prism_filament_painting.palette.sample_inputs sample`. A measurement batch uses CSV columns `spool,distance_mm,hex,date` with optional `lot,sample`.

## Optional tools

* `optimize` adds PyTorch for CPU height optimisation and user-supplied focus and height maps. On Linux and Windows, include the PyTorch CPU index when installing the extra, e.g. add `--extra-index-url https://download.pytorch.org/whl/cpu` to the pip install command. The launcher always selects CPU.
* `catalogues` adds the YAML reader. Use `python -m prism_filament_painting.palette.catalogues --help` for importing an explicitly supplied catalogue checkout into your library.
* `mcp` adds the stdio protocol for a local assistant client. Start `prism-palette-mcp --library sample/spools.csv` from your client's configuration. The three read-only tools list spools, rank palettes and render previews. Choose images you agree to share with your assistant and its provider.

Use `prism-calibrate board materials.json board` to generate four plate parts from four invented or user-supplied material records (each with id, hex and td_mm). The extract, fit and install subcommands process a photograph and its marked coordinates. Run `prism-calibrate --help` for details. The install subcommand accepts a directory override. For a workspace started with `--data-dir`, pass its calibration subdirectory. A synthetic capture is labelled separately from a measured print.

Prism includes its browser scripts and an original, hand-written stylesheet using system fonts. All image processing and print generation run locally.
