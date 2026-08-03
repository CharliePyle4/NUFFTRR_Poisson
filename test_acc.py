import sys
import pandas as pd
from Tests.CPU.testing_helpers import run_tests_pipeline

N_vals = [512]
M_vals = [512]
METHODS = [
    dict(name="Unif-FFT", label="Uniform / FFT", azu_unif=2, mesh_kind="uniform", solver_azu_unif=2, use_nudft=None),
    dict(name="Fixed-jittered-NUFFT", label="Fixed jittered / NUFFT", azu_unif=1, mesh_kind="jittered", solver_azu_unif=1, use_nudft=False),
]

print("Running Accuracy Verification...")
df = run_tests_pipeline(N_vals, M_vals, fixed_other=None, methods=METHODS, test_type="P1_Table1", mute=True)
print("\nResults:")
print(df)
