<!-- Slides for the training days -->

# Concurrency and Parallelism in Python

### Scientific Machine Learning - DAMSL

#### Georgios Arampatzis

<!-- <div style="margin-top: 50px;"></div>

<div style="margin-top: 50px;"></div> -->



---
## Why Concurrency matters in Scientific ML?

- Modern scientific workloads are **massively parallel**

- Data pipelines, simulations, training loops all repeat similar tasks

- Sequential Python underuses multi-core CPUs

- **Concurrency** lets multiple tasks make progress together

- **Goal**: structure code to exploit concurrency → improve throughput



---
## Concurrency vs Parallelism

- **Concurrency:** managing multiple tasks in overlapping time

- **Parallelism:** executing multiple tasks simultaneously

- Python supports both through:
    - **threading** →  concurrency
    - **multiprocessing** → true parallelism



---
## Threads vs Processes — Memory Model

- **Threads** live inside one process and share memory

- **Processes** run independently with isolated memory

- **Threads** communicate via shared objects → need synchronization

- **Processes** communicate via queues or pipes → safer, slower

- **CPython’s GIL** prevents true CPU parallelism in threads



---
## CPU-Bound vs I/O-Bound Tasks

- **CPU-bound:** performance limited by computation
    - e.g., matrix multiplication, numerical simulations

- **I/O-bound:** performance limited by waiting on external resources
    - e.g., disk access, network requests

- Choose threads or processes based on task type



--
## CPU-bound tasks

- **Definition:**
    - A CPU-bound task spends almost all its time performing calculations — arithmetic, loops, matrix ops, simulations.
    - The bottleneck is the processor’s computation speed.

- **Symptoms:**
    - CPU usage is near 100 % on one or more cores.
    - Adding more CPU cores can reduce runtime — if your code can use them.
    - I/O (disk, network) is minimal or absent.



--
## CPU-bound tasks

- **Examples in Scientific ML:**
    - Solving an ODE/PDE system numerically
    - Computing gradients manually in Python
    - Large matrix multiplications (if not vectorized)
    - Monte Carlo or parameter sweep simulations

- **Parallelism choice:**
    - Threads: don’t help (GIL prevents true parallelism)
    - Processes / native libraries: do help (each process or C thread runs independently)


--
## I/O-bound tasks

- **Definition:**
    - An I/O-bound task spends most of its time waiting for external resources — disk, network, or sensors — rather than doing arithmetic.
    - The bottleneck is data transfer speed or latency, not CPU power.

- **Symptoms:**
    - CPU usage is low even though the program feels “slow.”
    - Code often pauses on file reads, downloads, or database queries.



--
## I/O-bound tasks

- **Examples in Scientific ML:**
    - Loading millions of images or CSVs from disk
    - Fetching data from APIs or a remote dataset
    - Writing model checkpoints or logging during training
    - Streaming sensor or simulation output

- **Parallelism choice:**
    - Threads: help a lot — while one thread waits for I/O, another can run
    - Processes: can work too, but are heavier and usually unnecessary



---
## The GIL (Global Interpreter Lock)

- Ensures **only one thread runs Python** code at a time

- Protects **Python’s memory system** from corruption

- Limits parallel CPU use for “pure Python” code

- Libraries in C (NumPy, PyTorch) can release the GIL during heavy math so they achieve real multi-core performance

- **Why it exists:**
    - Python keeps track of how many times each object is used — this helps it know when to delete unused objects.
    - If multiple threads changed that information at the same time, Python could become confused and crash.
    - So, the GIL acts like a traffic light that allows only one thread to use Python’s internal memory system at a time.



---
## Where Threads and Processes Help in Sci ML

- **Threads:**
    - Data loading & preprocessing
        - DataLoader workers
    - I/O overlap
        - reading files
        - network fetches

- **Processes:**
    - Simulation ensembles, parameter sweeps
    - CPU-bound tasks
        - feature extraction
        - inference loops

- Combine both for hybrid workflows



---
# Threading



---
## Concurrency with Threads

- Threads = **multiple flows inside one process** (shared memory)

- Great for I/O-bound tasks (overlap waiting)

- Not for CPU speedups (GIL) — use processes for that

- Use high-level APIs: ThreadPoolExecutor



---
## Threading Basics

- Create → start → join (threading.Thread)

- Shared state needs protection (race conditions)

- Synchronization tools: Lock, RLock, Event

- Prefer executors for simplicity and safety



---
## ThreadPoolExecutor Patterns

1. **Two main usage patterns:**

    - **submit() + as_completed()**
        - fine-grained, flexible
        - Use this when tasks have uneven duration or when you want to process results as they finish.

    - **map()**
        - simple, ordered
        - Use this when all tasks are similar in size and you just need the results in input order.


2. **Choosing how many threads:**
``` Python
max_workers = min(32, os.cpu_count() + 4)
```



--
## ThreadPoolExecutor Patterns: Example structure

``` Python
from concurrent.futures import ThreadPoolExecutor, as_completed

def worker(x):  # lightweight task
    ...

# Using map (preserves order)
with ThreadPoolExecutor(max_workers=8) as ex:
    for result in ex.map(worker, range(10)):
        print(result)

# Using submit (results as they finish)
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(worker, i) for i in range(10)]
    for fut in as_completed(futures):
        print(fut.result())
```



---
# Multiprocessing



---
## Parallelism with Processes

- **True parallelism:** each worker = separate Python interpreter (own GIL)

- **Best for:** CPU-bound loops, simulations, numeric kernels

- **Trade-off:** higher startup + IPC (pickling) overhead

- **High-level API:** concurrent.futures.ProcessPoolExecutor



---
## Multiprocessing Basics

- **Separate memory spaces; communicate via pickled args/results**

- **Start methods:** spawn (macOS/Windows default), fork (Linux), forkserver (Unix)

- **Always guard entry:** if __name__ == "__main__":

- Target functions must be *module-level* (no lambdas/closures)

- Control nested native threads (e.g., OMP_NUM_THREADS=1) to *avoid oversubscription*



---
## ProcessPoolExecutor Patterns


- **map(fn, iterable, *, chunksize=...)** → simple, ordered results; great for uniform tasks

- **submit(fn, arg) + as_completed(futs)** → streaming, unordered; best for variable durations

- Make tasks coarse (≥50–100 ms CPU) to *amortize pickling/IPC*

- Use **chunksize** to batch many small items; keep result objects modest

- **Handle exceptions** via future.result(); optional timeouts for robustness

- **Reproducibility:** pass per-process seeds/indices explicitly



--
## ProcessPoolExecutor Patterns: Example structure

``` Python
# map(): ordered, simple
from concurrent.futures import ProcessPoolExecutor
def work(x): ...
if __name__ == "__main__":
    xs = [...]
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(work, xs, chunksize=1))

# submit()+as_completed(): streaming, flexible
from concurrent.futures import ProcessPoolExecutor, as_completed
def work(x): ...
if __name__ == "__main__":
    xs = [...]
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(work, x) for x in xs]
        for f in as_completed(futs):
            res = f.result()
            ...
```