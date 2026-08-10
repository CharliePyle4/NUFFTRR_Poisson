i import os, sys
import pandas as pd
import numpy as np

repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
os.chdir(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Tests.CPU.testing_helpers import (
    run_tests_pipeline,
    set_global_config,
)

BASE_MACHINE_PRECS = {
    "Unif-FFT": 0.0,
    "Unif-NUDFT": 1e-20,
    "Unif-NUFFT": 1e-20,
    "Fixed-jittered-NUDFT": 1e-20,
    "Fixed-jittered-NUFFT": 1e-20,
    "Fixed-stratified_rand-NUDFT": 1e-20,
    "Fixed-stratified_rand-NUFFT": 1e-20,
    "Fixed-rand-NUDFT": 1e-14,
    "Fixed-rand-NUFFT": 1e-12,
}

LAMBDAS_6_SWEEPS = {
    # -------------------------------------------------------------------------
    # 3 Sweeps Varying N (Fixed M = 32, 128, 512)
    # -------------------------------------------------------------------------
    "VaryN_M32": {
        **BASE_MACHINE_PRECS,
        "Fixed-clustered-NUDFT": {32: 1e-16, 64: 1e-12, 128: 1e-11, 256: 1e-12, 512: 1e-11},
        "Fixed-clustered-NUFFT": 1e-6,
        "Fixed-sine-NUDFT":      {32: 1e-16, 64: 1e-16, 128: 1e-12, 256: 1e-11, 512: 1e-08},
        "Fixed-sine-NUFFT":      {32: 1e-6, 64: 1e-12, 128: 5e-6, 256: 5e-5, 512: 1e-5},
    },
    "VaryN_M128": {
        **BASE_MACHINE_PRECS,
        "Fixed-clustered-NUDFT": {32: 1e-16, 64: 1e-12, 128: 1e-11, 256: 1e-12, 512: 1e-11},
        "Fixed-clustered-NUFFT": 1e-6,
        "Fixed-sine-NUDFT":      {32: 1e-16, 64: 1e-16, 128: 1e-12, 256: 1e-11, 512: 1e-08},
        "Fixed-sine-NUFFT":      {32: 1e-6, 64: 1e-12, 128: 5e-6, 256: 5e-5, 512: 1e-5},
    },
    "VaryN_M512": {
        **BASE_MACHINE_PRECS,
        "Fixed-clustered-NUDFT": {32: 1e-16, 64: 1e-12, 128: 1e-11, 256: 1e-12, 512: 1e-11},
        "Fixed-clustered-NUFFT": 1e-6,
        "Fixed-sine-NUDFT":      {32: 1e-16, 64: 1e-16, 128: 1e-12, 256: 1e-11, 512: 1e-08},
        "Fixed-sine-NUFFT":      {32: 1e-6, 64: 1e-12, 128: 5e-6, 256: 5e-5, 512: 1e-5},
    },

    # -------------------------------------------------------------------------
    # 3 Sweeps Varying M (Fixed N = 32, 128, 512)
    # -------------------------------------------------------------------------
    "VaryM_N32": {
        **BASE_MACHINE_PRECS,
        "Fixed-clustered-NUDFT": 1e-16,
        "Fixed-clustered-NUFFT": 1e-6,
        "Fixed-sine-NUDFT":      1e-16,
        "Fixed-sine-NUFFT":      1e-6,
    },
    "VaryM_N128": {
        **BASE_MACHINE_PRECS,
        "Fixed-clustered-NUDFT": 1e-11,
        "Fixed-clustered-NUFFT": 1e-6,
        "Fixed-sine-NUDFT":      1e-12,
        "Fixed-sine-NUFFT":      5e-6,
    },
    "VaryM_N512": {
        **BASE_MACHINE_PRECS,
        "Fixed-clustered-NUDFT": 1e-11,
        "Fixed-clustered-NUFFT": 1e-6,
        "Fixed-sine-NUDFT":      1e-08,
        "Fixed-sine-NUFFT":      1e-05,
    },
}

set_global_config(
    R=1.0,
    rad_unif=1,
    tol_nufft=1e-10,
    maxiter_nufft=200,
    eps_finufft=1e-15,
    precond_shift=1e-8,
    kde_oversample=4,
    kde_bandwidth=1.0,
    quad_rule=1,
    BC_choice=1,
    problem_type=0
)

N_vals_c = [32, 64, 128, 256, 512]
M_vals_c = [32, 64, 128, 256, 512]

METHODS_COMP = [
    dict(name="Unif-FFT", label="Uniform / FFT", azu_unif=2, mesh_kind="uniform", solver_azu_unif=2, use_nudft=None),
    dict(name="Unif-NUDFT", label="Uniform / NUDFT", azu_unif=2, mesh_kind="uniform", solver_azu_unif=1, use_nudft=True),
    dict(name="Unif-NUFFT", label="Uniform / NUFFT", azu_unif=2, mesh_kind="uniform", solver_azu_unif=1, use_nudft=False),
]
for kind in ("rand", "stratified_rand", "jittered", "clustered", "sine"):
    METHODS_COMP += [
        dict(name=f"Fixed-{kind}-NUDFT", label=f"Fixed {kind} / NUDFT", azu_unif=1, mesh_kind=kind, solver_azu_unif=1, use_nudft=True),
        dict(name=f"Fixed-{kind}-NUFFT", label=f"Fixed {kind} / NUFFT", azu_unif=1, mesh_kind=kind, solver_azu_unif=1, use_nudft=False),
    ]

FIXED_M_LIST = [32, 128, 512]
FIXED_N_LIST = [32, 128, 512]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Running 6 Independent Sweeps:")
for fixed_m in FIXED_M_LIST:
    sweep_key = f"VaryN_M{fixed_m}"
    reg_table = LAMBDAS_6_SWEEPS[sweep_key]
    df_cN = run_tests_pipeline(N_vals_c, None, fixed_other=fixed_m, methods=METHODS_COMP, test_type="Accuracy_VaryN", reg_param=reg_table, mute=True)
    piv = df_cN.pivot_table(index="N", columns="label", values="L2_rel")
    print(f"\n{'='*100}\n{sweep_key} (Fixed M={fixed_m}) Accuracy:\n{'='*100}")
    print(piv.map(lambda x: f"{x:.2e}"))

for fixed_n in FIXED_N_LIST:
    sweep_key = f"VaryM_N{fixed_n}"
    reg_table = LAMBDAS_6_SWEEPS[sweep_key]
    df_cM = run_tests_pipeline(None, M_vals_c, fixed_other=fixed_n, methods=METHODS_COMP, test_type="Accuracy_VaryM", reg_param=reg_table, mute=True)
    piv = df_cM.pivot_table(index="M", columns="label", values="L2_rel")
    print(f"\n{'='*100}\n{sweep_key} (Fixed N={fixed_n}) Accuracy:\n{'='*100}")
    print(piv.map(lambda x: f"{x:.2e}"))
