"""Aggregate runs/*.jsonl into one row per (label, toolset)."""
import json
import sys
from pathlib import Path

METRICS = ["file_acc1", "file_hit5", "file_all5", "func_hit5"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main(run_dir="runs"):
    print(f"{'label':<16}{'tools':<6}{'n':>4}" + "".join(f"{m:>11}" for m in METRICS)
          + f"{'submit%':>9}{'invalid/ep':>11}{'turns':>7}{'sec':>7}")
    for path in sorted(Path(run_dir).glob("*.jsonl")):
        recs = [json.loads(line) for line in path.open(encoding="utf-8")]
        if not recs:
            continue
        r0 = recs[0]
        row = f"{r0['label']:<16}{r0['toolset']:<6}{len(recs):>4}"
        row += "".join(f"{mean([r[m] for r in recs]):>11.3f}" for m in METRICS)
        row += f"{mean([r['end'] == 'submit' for r in recs]):>9.2f}"
        row += f"{mean([r['stats'].get('invalid_calls', 0) for r in recs]):>11.2f}"
        row += f"{mean([r['stats'].get('turns', 0) for r in recs]):>7.1f}"
        row += f"{mean([r['seconds'] for r in recs]):>7.0f}"
        print(row)


if __name__ == "__main__":
    main(*sys.argv[1:])
