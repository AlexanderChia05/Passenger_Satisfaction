"""Live training progress in the terminal.

    python train_status.py            # refresh every 10 s until the run ends (Ctrl+C to quit)
    python train_status.py --once     # print once
"""
import argparse
import time
from pathlib import Path

from src import config as C
from src.progress import fmt_secs, read_status


def render(out: Path) -> bool:
    overall, tasks = read_status(out)
    if overall is None:
        print("No run found in", out)
        return False
    now = time.time()
    print(f"\nSkyPulse training | status {overall['status'].upper()} | elapsed {fmt_secs(now - overall['started'])}"
          f" | remaining ~ {fmt_secs(overall.get('eta_s'))}")
    for name, st in overall["steps"].items():
        bar_n = int(20 * st["tasks_done"] / st["tasks_total"]) if st["tasks_total"] else (20 if st["status"] == "done" else 0)
        took = fmt_secs((st["end"] or now) - st["start"]) if st["start"] else "-"
        print(f"  {name:10s} {st['status']:8s} [{'#' * bar_n}{'.' * (20 - bar_n)}] "
              f"{st['tasks_done']:>3}/{st['tasks_total']:<3} took {took:>8}  est {fmt_secs(st['estimate_s'])}")
    for t in tasks:
        if t.get("kind") == "tuning":
            best = f"{t['best']:.4f}" if t.get("best") is not None else "-"
            print(f"    tuning {t['member']:4s} trial {t['done']}/{t['total']}  best F1 {best}  "
                  f"pruned {t.get('pruned', 0)}  elapsed {t.get('elapsed', '-')}  eta {t.get('eta', '-')}")
    return overall["status"] == "running"


def main():
    import sys
    sys.stdout.reconfigure(errors="replace")  # never crash on console encodings
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.ARTIFACTS_FULL))
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while render(Path(args.out)) and not args.once:
        time.sleep(10)


if __name__ == "__main__":
    main()
