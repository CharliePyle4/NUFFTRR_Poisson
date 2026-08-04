# 1. Install it if you don't have it: pip install threadpoolctl
from threadpoolctl import threadpool_info, threadpool_limits
import pprint
import numpy as np  # <--- Add this!
import scipy.linalg # <--- Add this!
import finufft

# 2. See exactly what math libraries are loaded and how many cores they are using
pprint.pprint(threadpool_info())


import os
import scipy.fft as sp_fft

print("\n--- SciPy FFT Threads ---")
# Unlike NumPy/SciPy lstsq which delegates to OpenBLAS, SciPy FFT uses its own 
# internal C++ pocketfft workers. When we pass workers=-1, it defaults to os.cpu_count()
print(f"When workers=-1, SciPy FFT will spawn exactly {os.cpu_count()} threads!")

# You can also manually see SciPy's current default worker configuration:
print(f"SciPy FFT global default worker count: {sp_fft.get_global_workers()}")


# 3. Force SciPy/NumPy to only use 4 cores for their dense matrix operations
with threadpool_limits(limits=4):
    # Any lstsq or dense matrix operations in this block will be capped at 4 cores
    # core = _invert_nufft_block_cgls_shared(...)
    pass
