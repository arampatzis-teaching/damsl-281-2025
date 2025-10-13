import numpy as np
import time
from multiprocessing import shared_memory


def fetch_data(task_id):
    """Simulate I/O-bound work: e.g., file read or download."""
    delay = np.random.uniform(0.2, 1.0)
    time.sleep(delay)
    return f"Task {task_id} done in {delay:.2f}s"


def cpu_heavy(n = 10_000_000) -> int:
    """
    CPU-heavy task that sums the squares of the first n integers modulo 97.
    """
    s = 0
    for i in range(n):
        s += (i * i) % 97
    return s


def chunk_sum(shm_name, shape, dtype_str, start, end):
    """
    Attach to shared memory by name and sum a slice [start:end].
    Args is a tuple: (shm_name, shape, dtype_str, start, end).
    Returns a Python float.
    """
    
    dtype = np.dtype(dtype_str)
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return float(np.sum(arr[start:end]))
    finally:
        shm.close()