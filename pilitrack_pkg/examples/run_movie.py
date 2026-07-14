"""Analyze one pili movie (ND2 / TIFF / OME-TIFF / CZI) end to end.

    python examples/run_movie.py "Labelled data/trial01007.nd2" --fast
    python examples/run_movie.py movie.ome.tif --out results --no-viewer
    python examples/run_movie.py m.nd2 --config results/config.json   # reuse settings

Writes pili.csv, cells.csv, QC overlays, config.json and a reproducibility
manifest.json to --out. All logic lives in pilitrack.analyze (importable/tested).
"""
from pilitrack.analyze import main

if __name__ == "__main__":
    main()
