# Getting pilitrack running (from zero)

You need this once per computer. After it, you just double-click to open the tool.

## Windows

1. **Install Miniconda** (skip if you already have Anaconda/Miniconda):
   download from <https://docs.conda.io/en/latest/miniconda.html> and run the
   installer with the defaults.
2. **Double-click `Install pilitrack.bat`** in this folder. It installs
   everything (a few minutes the first time). Wait for **"All set!"**.
3. **Double-click `Start pilitrack (browser)`** — the tool opens in your browser.
   (Or `Start pilitrack (desktop)` for the hand-labeling window.)

## macOS / Linux

1. Install Miniconda (same link as above).
2. Double-click **`Install pilitrack.command`** (first time: right-click → Open).
3. Double-click **`Start pilitrack (browser).command`**.

## Prefer a clean, isolated environment?

```bash
conda env create -f environment.yml     # makes a "pilitrack" env with everything
conda activate pilitrack
pilitrack-web                            # browser app   (or: pilitrack-gui)
```

## Just pip, no conda?

```bash
python -m venv .venv
.venv\Scripts\activate                   # Windows  (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
pilitrack-web
```

## Big movies (500 MB+)

The tool never needs the whole movie in memory:

- **In the browser app:** type the **File path** (don't use *…or upload*) — the
  file stays on disk and only the part you're viewing is read. "Fast preview"
  (on by default) loads just a centre crop of the first frames; the **Downsample**
  slider (2× / 4×) shrinks the whole field to fit. (Uploads are also allowed up to
  8 GB, but the path is lighter.)
- **In scripts / desktop:** use `--fast`, or `--roi Y0 Y1 X0 X1` / `--frames S E`
  to bound what's loaded.

## Optional add-ons

- **Zeiss CZI movies:** `pip install czifile`
- **Touching-cell segmentation** (accurate per-cell counts): `pip install cellpose`

## What each launcher does

| File | Opens |
|---|---|
| `Install pilitrack` | installs the tool + all dependencies (run once) |
| `Start pilitrack (browser)` | the browser app — pick a movie, analyze, download results |
| `Start pilitrack (desktop)` | the napari window — trace missed pili, edit cells |

## If a launcher doesn't work

Open an **Anaconda Prompt** (Windows) or a terminal (Mac/Linux) and run the
command directly, e.g. `pilitrack-web`. The error it prints tells us what's
missing — send it over and we'll sort it.

Data note: your movies never leave your computer. The raw `.nd2` files aren't in
this folder's version control (they're large) — keep/share them separately.
