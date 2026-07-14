"""`pilitrack-web` — open the browser app.

Starts a local Streamlit server and opens pilitrack in your default browser.
Everything stays on this machine; nothing is uploaded anywhere.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv=None):
    webapp = Path(__file__).with_name("webapp.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(webapp),
           "--browser.gatherUsageStats", "false"]
    extra = list(sys.argv[1:] if argv is None else argv)
    cmd += extra
    try:
        raise SystemExit(subprocess.call(cmd))
    except FileNotFoundError:  # pragma: no cover - streamlit missing
        print("Streamlit is not installed. Run:  pip install streamlit")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
