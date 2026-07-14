"""Run the full pipeline on a synthetic movie and compare to ground truth.

    python examples/run_synthetic.py
"""
import numpy as np

from pilitrack.config import AcquisitionConfig
from pilitrack.pipeline import analyze_movie
from pilitrack.synth import make_movie

TRUE_V_EXT, TRUE_V_RET, TRUE_MAX = 500.0, 500.0, 2000.0
TRUE_PILIATED_FRACTION = 0.8

cfg = AcquisitionConfig(dt_s=0.5, pixel_size_nm=65.0, min_pilus_length_nm=200.0)
mov = make_movie(cfg, n_cells=8, shape=(200, 200),
                 piliated_fraction=TRUE_PILIATED_FRACTION,
                 v_ext_nm_s=TRUE_V_EXT, v_ret_nm_s=TRUE_V_RET,
                 max_length_nm=TRUE_MAX, n_cycles=2, seed=7)

res = analyze_movie(mov.stack, mov.cell_stack, cfg)
pop, cells, pili = res["population"], res["cell"], res["pilus"]

print("=== population ===")
print(f"cells detected      : {pop['n_cells']}")
print(f"percent piliated    : {pop['percent_piliated']:.0f}%  "
      f"(ground truth ~{TRUE_PILIATED_FRACTION*100:.0f}%)")

print("\n=== per-pilus kinetics (recovered) ===")
if not pili.empty:
    print(pili[["cell_id", "max_length_nm",
                "mean_extension_velocity_nm_s",
                "mean_retraction_velocity_nm_s"]].round(0).to_string(index=False))
    print(f"\nmedian extension velocity : "
          f"{np.nanmedian(pili['mean_extension_velocity_nm_s']):.0f} nm/s "
          f"(truth {TRUE_V_EXT:.0f})")
    print(f"median retraction velocity: "
          f"{np.nanmedian(pili['mean_retraction_velocity_nm_s']):.0f} nm/s "
          f"(truth {TRUE_V_RET:.0f})")
    print(f"median max length         : "
          f"{np.nanmedian(pili['max_length_nm']):.0f} nm (truth {TRUE_MAX:.0f})")
