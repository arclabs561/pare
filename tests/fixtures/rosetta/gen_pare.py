# /// script
# requires-python = ">=3.10"
# dependencies = ["pymoo", "numpy"]
# ///
"""Rosetta fixture generator for pare hypervolume.

Provenance for pare_hypervolume.json. The reference hypervolume comes from
pymoo's HV indicator (the standard multi-objective benchmark library). Both pare
and pymoo compute the exact Lebesgue measure of the region dominated by the front
and bounded by the reference point, under the MINIMIZE convention, so they agree
exactly (EXACT class, both f64).

pare orients each objective internally (Minimize: ref - value), so an all-Minimize
ParetoFrontier with ref_point = nadir matches pymoo HV(ref_point) directly. Both
filter dominated points, so passing the raw point set is fine.

Regenerate: uv run tests/fixtures/rosetta/gen_pare.py
"""

import json
import platform
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV

SEED = 0
rng = np.random.default_rng(SEED)

# 2D and 3D minimization point sets in [0, 1]; ref point is "worse than all".
points_2d = rng.uniform(0.0, 1.0, size=(8, 2))
ref_2d = [1.1, 1.1]
hv_2d = float(HV(ref_point=np.array(ref_2d))(points_2d))

points_3d = rng.uniform(0.0, 1.0, size=(6, 3))
ref_3d = [1.1, 1.1, 1.1]
hv_3d = float(HV(ref_point=np.array(ref_3d))(points_3d))

fixture = {
    "provenance": {
        "generator": "gen_pare.py",
        "library": "pymoo",
        "pymoo_version": __import__("pymoo").__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "seed": SEED,
        "note": "minimize convention; both compute exact hypervolume in f64.",
    },
    "points_2d": points_2d.tolist(),
    "ref_2d": ref_2d,
    "points_3d": points_3d.tolist(),
    "ref_3d": ref_3d,
    "expected": {
        "hv_2d": hv_2d,
        "hv_3d": hv_3d,
    },
}

out = Path(__file__).parent / "pare_hypervolume.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
print(f"hv_2d {hv_2d:.12f}")
print(f"hv_3d {hv_3d:.12f}")
print(f"wrote {out}")
