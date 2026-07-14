"""Batch-analyze a whole folder of pili movies with one consistent config.

    python examples/run_batch.py "Labelled data/" --out batch_results
    python examples/run_batch.py data/ --config batch_results/some/config.json --recursive

Produces one result folder per movie plus a combined summary.csv, pili_all.csv,
and batch_manifest.json. All logic lives in pilitrack.batch (importable/tested).
"""
from pilitrack.batch import main

if __name__ == "__main__":
    main()
