import os
import sys
repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from Tests.CPU.testing_helpers import run_tests_pipeline, render_accuracy

# Test Problem 0 Table 1
print("Running test of Problem 0 Table 1...")
res = run_tests_pipeline(
    problem_type=0,
    test_type=0,
    N_values=[32, 64],
    M_values=[32, 64],
    N_fixed=32,
    M_fixed=64,
    nonunif_type="jittered",
    mute_output=False
)
print("Results:")
for k, v in res.items():
    print(k, "L2 errors:", v["l2_rel_error"])
