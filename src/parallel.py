"""Process-parallel task runner for training.

Each worker process loads the training data once (initializer), so tasks only
carry row indices and an estimator. Nested parallelism is controlled: every
estimator's n_jobs is set to `inner_jobs` inside the worker, so
workers x inner_jobs ~= number of CPU threads (no oversubscription).
Tasks are submitted longest-first (LPT scheduling) to finish the step sooner.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_DATA = {}


def _init(inner_jobs, X, y, out):
    os.environ["SKYPULSE_INNER_JOBS"] = str(inner_jobs)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(inner_jobs)
    import warnings
    warnings.filterwarnings("ignore")
    _DATA.update(X=X, y=np.asarray(y), out=out, inner=inner_jobs)


def data():
    return _DATA["X"], _DATA["y"]


def out_dir():
    return _DATA["out"]


def set_inner_jobs(est, n=None):
    """Set every nested n_jobs parameter of an estimator (pipelines, bagging, voting, xgb)."""
    n = n if n is not None else _DATA.get("inner", 1)
    params = {k: n for k in est.get_params(deep=True) if k == "n_jobs" or k.endswith("__n_jobs")}
    if params:
        est.set_params(**params)
    return est


def run(tasks, workers, inner_jobs, X, y, out, tracker=None, step=None):
    """tasks: list of dicts {name, fn, args, cost}. Returns {name: result or None}."""
    results = {}
    tasks = sorted(tasks, key=lambda t: -t.get("cost", 1))
    if tracker and step:
        tracker.add_tasks(step, len(tasks))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(inner_jobs, X, y, out)) as pool:
        futures = {pool.submit(_timed, t["fn"], *t["args"]): t["name"] for t in tasks}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                res, secs = fut.result()
                results[name] = res
                failed = res is None
            except Exception as exc:  # worker crashed: record and keep the other tasks
                print(f"  [{name}] failed: {type(exc).__name__}: {exc}", flush=True)
                results[name], secs, failed = None, 0.0, True
            if tracker and step:
                tracker.task_done(step, name, secs, failed=failed)
    return results


def _timed(fn, *args):
    t0 = time.perf_counter()
    return fn(*args), time.perf_counter() - t0
