import finufft
import threadpoolctl
import os

print(f"--- Thread Count Diagnostics ---")

# Check System Threads (FINUFFT defaults to all cores)
print(f"FINUFFT (NUFFT) is using: {os.cpu_count()} threads (Default OpenMP behavior)")

# Check SciPy/NumPy BLAS/LAPACK (NUDFT) threads
try:
    pools = threadpoolctl.threadpool_info()
    for pool in pools:
        print(f"BLAS/LAPACK Backend ({pool.get('user_api', 'Unknown')} - {pool.get('filepath', '').split(os.sep)[-1]}): {pool.get('num_threads')} threads")
except Exception as e:
    print(f"Could not read BLAS threads via threadpoolctl: {e}")
