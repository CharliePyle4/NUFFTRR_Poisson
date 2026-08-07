import json

nb_path = r"c:\Users\charl\NUFFTRR_Poisson\Tests\CPU\testing.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 1: Imports
nb["cells"][1]["source"] = [
    "import os, sys\n",
    "import pandas as pd\n",
    "# Main project root\n",
    "repo_root = r\"c:\\Users\\charl\\NUFFTRR_Poisson\"\n",
    "os.chdir(repo_root)\n",
    "if repo_root not in sys.path:\n",
    "    sys.path.insert(0, repo_root)\n",
    "\n",
    "from Tests.CPU.testing_helpers import (\n",
    "    run_tests_pipeline,\n",
    "    set_global_config,\n",
    "    get_global_config,\n",
    "    render_accuracy,\n",
    "    render_runtime,\n",
    "    render_table2_accuracy,\n",
    "    render_table2_runtime,\n",
    "    plot_accuracy_table1,\n",
    "    plot_runtime_table1,\n",
    "    plot_accuracy_table2,\n",
    "    plot_runtime_table2,\n",
    "    plot_accuracy_comparison,\n",
    "    plot_runtime_comparison\n",
    ")\n"
]

# Cell 2: Markdown header
nb["cells"][2]["source"] = [
    "# Setup & Configurable Parameters\n",
    "Configure problem equations, solver tolerances, Tikhonov regularization $\\lambda$, and grid resolutions."
]

# Cell 3: Configurable Setup Code
nb["cells"][3]["source"] = [
    "# =============================================================================\n",
    "# Domain & Radial Grid Settings\n",
    "# =============================================================================\n",
    "R = 1.0                     # Disk radius\n",
    "RAD_UNIF = 1                # 1: Uniform radial grid\n",
    "\n",
    "# =============================================================================\n",
    "# Solver Tolerances & Regularization (Lambda)\n",
    "# =============================================================================\n",
    "TOL_NUFFT = 1e-10           # CG residual tolerance for NUFFT (e.g. 1e-8, 1e-10, 1e-14)\n",
    "MAXITER_NUFFT = 200         # Maximum CG iterations\n",
    "REG_PARAM = 1e-12           # Tikhonov regularization lambda / condition threshold\n",
    "QUAD_RULE = 1               # 1: Trapezoidal, 2: Simpson's rule\n",
    "BC_CHOICE = 1               # 1: Dirichlet, 2: Neumann\n",
    "\n",
    "# =============================================================================\n",
    "# Problem Selection / Equation Setup\n",
    "# Options:\n",
    "#   0 or 1: Borges & Daripa Problem 1 (Polynomial-Exponential)\n",
    "#   2: Problem 2 (Trigonometric)\n",
    "#   \"sharp_wave\": Sharp localized wave packet on disk\n",
    "#   \"custom\": Custom user dict with keys {'u', 'f', 'g_dir', 'g_neu'}\n",
    "# =============================================================================\n",
    "PROBLEM_TYPE = 0\n",
    "CUSTOM_PROBLEM = None\n",
    "\n",
    "# Apply global configuration to testing engine\n",
    "set_global_config(\n",
    "    R=R,\n",
    "    rad_unif=RAD_UNIF,\n",
    "    tol_nufft=TOL_NUFFT,\n",
    "    maxiter_nufft=MAXITER_NUFFT,\n",
    "    reg_param=REG_PARAM,\n",
    "    quad_rule=QUAD_RULE,\n",
    "    BC_choice=BC_CHOICE,\n",
    "    problem_type=PROBLEM_TYPE,\n",
    "    custom_problem=CUSTOM_PROBLEM\n",
    ")\n",
    "\n",
    "# =============================================================================\n",
    "# Grid Sweep Resolutions\n",
    "# =============================================================================\n",
    "N_vals_p0 = [32, 64, 128, 256, 512]\n",
    "M_vals_p0 = [32, 64, 128, 256, 512]\n",
    "N_fixed_p0 = 32\n",
    "\n",
    "N_vals_p1 = [32, 64, 128, 256, 512]\n",
    "M_vals_p1 = [32, 64, 128, 256, 512]\n",
    "N_fixed_p1 = 32\n",
    "\n",
    "N_vals_p2 = [32, 64, 128, 256, 512]\n",
    "M_vals_p2 = [32, 64, 128, 256, 512]\n",
    "N_fixed_p2 = 32\n",
    "\n",
    "N_vals_c = [32, 64, 128, 256, 512]\n",
    "M_vals_c = [32, 64, 128, 256, 512]\n",
    "\n",
    "MUTE_OUTPUT = True\n"
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated testing.ipynb setup cell successfully!")
