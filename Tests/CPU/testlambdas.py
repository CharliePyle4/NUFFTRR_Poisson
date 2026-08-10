import os
import sys
import pandas as pd

# -----------------------------------------------------------------------------
# Repo setup
# -----------------------------------------------------------------------------
repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
os.chdir(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Tests.CPU.testing_helpers import (
    run_tests_pipeline,
    set_global_config,
    get_global_config,
)

# -----------------------------------------------------------------------------
# User‑configurable parameters
# -----------------------------------------------------------------------------
# Single resolution
N_SINGLE = 256          # angular resolution N
M_SINGLE = 512          # radial resolution M

# Mesh type: "uniform", "rand", "stratified_rand", "jittered", "clustered", "sine"
MESH_KIND = "rand"

# Solver choice: True => NUDFT, False => NUFFT
USE_NUDFT = True

# Grid of lambda values to test (interpreted according to your solver)
LAMBDA_GRID = [
    0.0,
    1e-25,
    1e-20,
    1e-18,
    1e-16,
    1e-14,
    1e-12,
    1e-10,
]

# Problem / solver defaults (adjust if needed)
R = 1.0
RAD_UNIF = 1
PROBLEM_TYPE = 0
CUSTOM_PROBLEM = None

TOL_NUFFT = 1e-10
MAXITER_NUFFT = 200
EPS_FINUFFT = 1e-20
PRECOND_SHIFT = 1e-8
KDE_OVERSAMPLE = 2
KDE_BANDWIDTH = 1.0
QUAD_RULE = 1
BC_CHOICE = 1

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def find_error_column(df: pd.DataFrame) -> str:
    """
    Heuristically find the error column in the pipeline DataFrame.
    Adjust candidate list if your schema is different.
    """
    candidates = [
        "L2_rel",       # <- your actual column
        "error",
        "rel_error",
        "relative_error",
        "L2_error",
        "l2_error",
        "err",
        "accuracy",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not find an error column. Available columns: {list(df.columns)}"
    )


def build_methods(mesh_kind: str, use_nudft: bool):
    """
    Build a single-method list compatible with run_tests_pipeline.
    """
    if mesh_kind == "uniform":
        name = "Unif-NUDFT" if use_nudft else "Unif-NUFFT"
        azu_unif = 2   # uniform grid
    else:
        name = f"Fixed-{mesh_kind}-NUDFT" if use_nudft else f"Fixed-{mesh_kind}-NUFFT"
        azu_unif = 1   # shared nonuniform mesh

    methods = [
        dict(
            name=name,
            label=name.replace("-", " "),
            azu_unif=azu_unif,
            mesh_kind=mesh_kind,
            solver_azu_unif=1,
            use_nudft=True if use_nudft else False,
        )
    ]
    return methods


def sweep_lambdas(
    lambdas,
    N,
    M,
    mesh_kind,
    use_nudft,
    mute=True,
):
    methods = build_methods(mesh_kind, use_nudft)
    results = []

    for lam in lambdas:
        set_global_config(
            R=R,
            rad_unif=RAD_UNIF,
            tol_nufft=TOL_NUFFT,
            maxiter_nufft=MAXITER_NUFFT,
            reg_param=lam,
            eps_finufft=EPS_FINUFFT,
            precond_shift=PRECOND_SHIFT,
            kde_oversample=KDE_OVERSAMPLE,
            kde_bandwidth=KDE_BANDWIDTH,
            quad_rule=QUAD_RULE,
            BC_choice=BC_CHOICE,
            problem_type=PROBLEM_TYPE,
            custom_problem=CUSTOM_PROBLEM,
        )

        df = run_tests_pipeline(
            [N],        # N_vals
            None,       # M_vals
            M,          # fixed_other
            methods,
            "Accuracy_VaryN",
            mute,
        )

        err_col = find_error_column(df)
        method_name = methods[0]["name"]
        row = df[(df["name"] == method_name) & (df["N"] == N)]
        if row.empty:
            raise RuntimeError(
                f"No row found for method={method_name}, N={N} in pipeline output."
            )

        error_value = float(row.iloc[0][err_col])
        results.append((lam, error_value))

    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Lambda sweep for a single (N, M) and mesh/solver choice")
    print(f"  N = {N_SINGLE}, M = {M_SINGLE}")
    print(f"  mesh_kind = {MESH_KIND}")
    print(f"  solver    = {'NUDFT' if USE_NUDFT else 'NUFFT'}")
    print("  lambdas   =", LAMBDA_GRID)
    print("\nRunning...\n")

    sweep = sweep_lambdas(
        lambdas=LAMBDA_GRID,
        N=N_SINGLE,
        M=M_SINGLE,
        mesh_kind=MESH_KIND,
        use_nudft=USE_NUDFT,
        mute=True,
    )

    print("=" * 79)
    print(f"Errors for {MESH_KIND} / {'NUDFT' if USE_NUDFT else 'NUFFT'} at N={N_SINGLE}, M={M_SINGLE}")
    print("=" * 79)
    print(f"{'lambda':>14}    {'error':>14}")
    print("-" * 79)
    for lam, err in sweep:
        print(f"{lam:14.6e}    {err:14.6e}")