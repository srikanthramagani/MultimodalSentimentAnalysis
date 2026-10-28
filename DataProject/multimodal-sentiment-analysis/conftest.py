import sys
import os

# Limit OpenBLAS and OpenMP thread pool allocation to prevent memory allocation exhaustion
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Ensure the project root is on the path so `app` is importable when running
# pytest from any working directory.
sys.path.insert(0, os.path.dirname(__file__))

import platform
# Prevent Python 3.12 Windows COM/WMI memory crash during torch initialization
platform.machine = lambda: "AMD64"

