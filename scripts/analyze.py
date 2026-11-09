"""Paired-bootstrap analysis of the grid: per-cell metrics with CIs, toolset
gains per label, and the RQ2 interaction (does the graph-tool gain grow as the
model shrinks?). Pure stdlib. Usage: python scripts/analyze.py runs [out.json]
"""
import glob
import json
import random
import sys
from collections import defaultdict

LABELS = ["12b-q4", "e4b-q4", "e2b-bf16", "e2b-q8", "e2b-q4"]  # capability order
TOOLSETS = ["A", "B", "C"]
METRIC = "file_hit5"
DRAWS = 10_000


def load(runs_dir):
    cells = defaultdict(dict)  # (label, ts) -> {instance_id: 0/1}
    extra = defaultdict(lambda: defaultdict(float))
    for path in glob.glob(f"{runs_dir}/*.jsonl"):
        for line in open(path, encoding="utf-8"):
            ep = json.loads(line)
            key = (ep["label"], ep["toolset"])
            cells[key][ep["instance_id"]] = 1.0 if ep[METRIC] else 0.0
            e = extra[key]
            e["n"] += 1
            e["submit"] += ep["end"] in ("submit", "forced_submit")
            e["invalid"] += ep["stats"].get("invalid_calls", 0) + ep["stats"].get("unknown_tool", 0)
            e["func_hit5"] += 1.0 if ep.get("func_hit5") else 0.0
    return cells, extra


def boot_mean(vals, rng):
    n = len(vals)
    out = []
    for _ in range(DRAWS):
        out.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    out.sort()
    return out[int(0.025 * DRAWS)], out[int(0.975 * DRAWS)]


def boot_delta(xs, ys, ids, rng):
    """Paired bootstrap of mean(ys) - mean(xs) over shared instances."""
    pairs = [(xs[i], ys[i]) for i in ids]
    n = len(pairs)
    deltas = []
    for _ in range(DRAWS):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        deltas.append(sum(b - a for a, b in s) / n)
    deltas.sort()
    lo, hi = deltas[int(0.025 * DRAWS)], deltas[int(0.975 * DRAWS)]
    p = sum(d <= 0 for d in deltas) / DRAWS  # one-sided: gain > 0
    return sum(ys[i] - xs[i] for i in ids) / n, lo, hi, p


def main():
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "analysis.json"
    rng = random.Random(0)
    cells, extra = load(runs_dir)
    result = {"metric": METRIC, "cells": {}, "gains": {}, "interaction": {}}

    print(f"== {METRIC} per cell (95% bootstrap CI) ==")
    for label in LABELS:
        for ts in TOOLSETS:
            vals = list(cells.get((label, ts), {}).values())
            if not vals:
                continue
            m = sum(vals) / len(vals)
            lo, hi = boot_mean(vals, rng)
            e = extra[(label, ts)]
            result["cells"][f"{label}/{ts}"] = {
                "mean": m, "lo": lo, "hi": hi, "n": len(vals),
                "submit": e["submit"] / e["n"], "invalid_per_ep": e["invalid"] / e["n"],
                "func_hit5": e["func_hit5"] / e["n"],
            }
            print(f"{label:9s} {ts}  {m:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"submit%={e['submit']/e['n']:.2f} invalid/ep={e['invalid']/e['n']:.2f}")

    print(f"\n== toolset gains per label (paired bootstrap, one-sided p: gain<=0) ==")
    for label in LABELS:
        a = cells.get((label, "A"), {})
        for ts in ("B", "C"):
            b = cells.get((label, ts), {})
            ids = sorted(set(a) & set(b))
            if not ids:
                continue
            d, lo, hi, p = boot_delta(a, b, ids, rng)
            result["gains"][f"{label}/{ts}-A"] = {"delta": d, "lo": lo, "hi": hi, "p": p}
            print(f"{label:9s} {ts}-A  {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}")

    # RQ2 interaction: is the (B-A) gain larger on the weakest model than the strongest?
    print("\n== RQ2 interaction: (B-A) at weak model minus (B-A) at strong model ==")
    for weak in ("e2b-q4", "e2b-q8"):
        for strong in ("12b-q4", "e4b-q4"):
            aw, bw = cells.get((weak, "A"), {}), cells.get((weak, "B"), {})
            as_, bs = cells.get((strong, "A"), {}), cells.get((strong, "B"), {})
            ids = sorted(set(aw) & set(bw) & set(as_) & set(bs))
            if not ids:
                continue
            diffs = [(bw[i] - aw[i]) - (bs[i] - as_[i]) for i in ids]
            n = len(diffs)
            boots = []
            for _ in range(DRAWS):
                boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
            boots.sort()
            lo, hi = boots[int(0.025 * DRAWS)], boots[int(0.975 * DRAWS)]
            p = sum(d <= 0 for d in boots) / DRAWS
            key = f"({weak} B-A) - ({strong} B-A)"
            result["interaction"][key] = {"delta": sum(diffs) / n, "lo": lo, "hi": hi, "p": p}
            print(f"{key}: {sum(diffs)/n:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.4f}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
