"""Score automated detection against hand-labeled frames.

    # 1. in the GUI, FULLY label a few frames and Save annotations.json
    # 2. then:
    python examples/run_validate.py "Labelled data/trial01007.nd2" \
        --labels annotations.json --roi 700 1212 700 1212

Reports detection precision/recall/F1 and length agreement (bias + 95% limits)
and writes validation.json. Same code path as the `pilitrack-validate` command.
Labels must be a *complete* tracing of the scored frames, made on the SAME ROI.
"""
from pilitrack.validate import main

if __name__ == "__main__":
    main()
