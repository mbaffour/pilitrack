"""Back-compat shim — ND2 analysis now goes through the general runner.

Use ``run_movie.py`` (any format) going forward; this forwards to the same code
path so existing commands keep working:

    python examples/run_nd2.py "Labelled data/trial01007.nd2" --fast
"""
from pilitrack.analyze import main

if __name__ == "__main__":
    main()
