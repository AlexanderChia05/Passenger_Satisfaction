"""Training progress tracker.

The main process writes  <out>/progress/overall.json  (steps, tasks, ETA) and
appends to <out>/progress/train.log. Worker processes write one small file per
running task in <out>/progress/tasks/ (e.g. tuning trials done). The Streamlit
Experiment lab and `python train_status.py` read these files live.
"""
import json
import os
import time
from pathlib import Path


def _write(path: Path, obj: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)  # atomic: readers never see a half-written file


def fmt_secs(s) -> str:
    if s is None:
        return "-"
    s = int(max(s, 0))
    h, m = divmod(s // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s % 60:02d}s"


class Tracker:
    def __init__(self, out: Path, steps: list[str], estimates: dict[str, float]):
        self.dir = out / "progress"
        (self.dir / "tasks").mkdir(parents=True, exist_ok=True)
        for f in (self.dir / "tasks").glob("*.json"):
            f.unlink()
        self.state = {"started": time.time(), "finished": None, "current": None, "status": "running",
                      "steps": {s: {"status": "pending", "estimate_s": estimates.get(s), "start": None, "end": None,
                                    "tasks_total": 0, "tasks_done": 0, "tasks_failed": 0} for s in steps}}
        self._save()

    # ---- step level ----
    def start_step(self, step, n_tasks=0):
        st = self.state["steps"][step]
        st.update(status="running", start=time.time(), tasks_total=n_tasks, tasks_done=0, tasks_failed=0)
        self.state["current"] = step
        self.log(f"START {step} ({n_tasks} tasks, estimate {fmt_secs(st['estimate_s'])})")
        self._save()

    def add_tasks(self, step, n):
        self.state["steps"][step]["tasks_total"] += n
        self._save()

    def task_done(self, step, name, seconds, failed=False):
        st = self.state["steps"][step]
        st["tasks_done"] += 1
        st["tasks_failed"] += int(failed)
        self.log(f"  {'FAIL' if failed else 'done'} {name} ({fmt_secs(seconds)}) "
                 f"[{st['tasks_done']}/{st['tasks_total']}]  step ETA {fmt_secs(self.step_eta(step))}")
        self._save()

    def end_step(self, step, failed=False):
        st = self.state["steps"][step]
        st.update(status="failed" if failed else "done", end=time.time())
        self.log(f"END {step} in {fmt_secs(st['end'] - st['start'])}")
        self._save()

    def finish(self, status="done"):
        self.state.update(status=status, finished=time.time(), current=None)
        self.log(f"RUN {status.upper()} in {fmt_secs(self.state['finished'] - self.state['started'])}")
        self._save()

    # ---- ETA ----
    def step_eta(self, step):
        st = self.state["steps"][step]
        if not st["start"]:
            return st["estimate_s"]
        elapsed = time.time() - st["start"]
        if st["tasks_done"] and st["tasks_total"]:
            return elapsed / st["tasks_done"] * (st["tasks_total"] - st["tasks_done"])
        return max((st["estimate_s"] or 0) - elapsed, 0)

    def overall_eta(self):
        total = 0.0
        for name, st in self.state["steps"].items():
            if st["status"] == "running":
                total += self.step_eta(name) or 0
            elif st["status"] == "pending":
                total += st["estimate_s"] or 0
        return total

    # ---- io ----
    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.dir / "train.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _save(self):
        self.state["updated"] = time.time()
        self.state["eta_s"] = self.overall_eta()
        _write(self.dir / "overall.json", self.state)


def report_task(out: Path, name: str, **info):
    """Called from worker processes: live status of one running task."""
    _write(out / "progress" / "tasks" / f"{name}.json", {"name": name, "updated": time.time(), **info})


def clear_task(out: Path, name: str):
    p = out / "progress" / "tasks" / f"{name}.json"
    if p.exists():
        p.unlink()


def read_status(out: Path):
    """(overall dict or None, list of running task dicts) for monitors."""
    d = out / "progress"
    if not (d / "overall.json").exists():
        return None, []
    overall = json.loads((d / "overall.json").read_text(encoding="utf-8"))
    tasks = []
    for f in sorted((d / "tasks").glob("*.json")):
        try:
            tasks.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue  # file being replaced right now
    return overall, tasks
