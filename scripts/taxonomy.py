"""Failure taxonomy over the grid trajectories (RQ3). Pure stdlib.

For every episode with file_hit5 == 0, assigns one primary failure mode:
  no_answer    - never produced a usable file list (files empty)
  forced_wrong - only answered when forced at the turn limit, and missed
  wrong_files  - submitted on its own, but the gold file is not in the top 5
and records cross-cutting diagnostics:
  gold_in_issue  - gold path is quoted in the issue text (localization given away)
  gold_in_map    - gold path appears in the toolset-C repo map (system prompt)
  gold_seen      - gold path appeared in an observation during the episode
  near_miss      - a submitted file sits in the same directory as a gold file
  invalid_frac   - (invalid + unknown-tool calls) / turns
  budget_bound   - mean completion tokens/turn >= 950 (reply cap is 1024:
                   thinking ate the budget before a tool call could be emitted)

Usage: python scripts/taxonomy.py runs [taxonomy.json] [failures.csv]
"""
import csv
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

LABELS = ["12b-q4", "e4b-q4", "e2b-bf16", "e2b-q8", "e2b-q4"]
TOOLSETS = ["A", "B", "C"]
SAMPLE_PER_CELL = 4  # stratified hand-labeling sample
BUDGET_TOKENS = 950


def classify(ep):
    if not ep["files"]:
        return "no_answer"
    if ep["end"] == "forced_submit":
        return "forced_wrong"
    return "wrong_files"


def diagnostics(ep):
    msgs = ep["messages"]
    golds = ep["gold_files"]
    system = msgs[0]["content"] if msgs else ""
    issue = msgs[1]["content"] if len(msgs) > 1 else ""
    later = "\n".join(m["content"] for m in msgs[2:] if m["role"] == "user")
    st = ep["stats"]
    turns = max(st.get("turns", 0), 1)
    gold_dirs = {os.path.dirname(g) for g in golds}
    return {
        "gold_in_issue": any(g in issue for g in golds),
        "gold_in_map": any(g in system for g in golds),
        "gold_seen": any(g in later for g in golds),
        "near_miss": any(os.path.dirname(str(f).replace("\\", "/")) in gold_dirs
                         for f in ep["files"]),
        "invalid_frac": round((st.get("invalid_calls", 0) + st.get("unknown_tool", 0)) / turns, 3),
        "budget_bound": st.get("completion_tokens", 0) / turns >= BUDGET_TOKENS,
        "turns": st.get("turns", 0),
        "comp_per_turn": round(st.get("completion_tokens", 0) / turns, 0),
    }


def main():
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    out_json = sys.argv[2] if len(sys.argv) > 2 else "taxonomy.json"
    out_csv = sys.argv[3] if len(sys.argv) > 3 else "failures.csv"

    episodes = []
    for path in glob.glob(f"{runs_dir}/*.jsonl"):
        for line in open(path, encoding="utf-8"):
            episodes.append(json.loads(line))

    rows = []
    agg = defaultdict(Counter)
    diag_sum = defaultdict(Counter)
    for ep in episodes:
        key = (ep["label"], ep["toolset"])
        agg[key]["n"] += 1
        d = diagnostics(ep)
        if ep["file_hit5"]:
            agg[key]["hit"] += 1
            diag_sum[key]["hit_gold_seen"] += d["gold_seen"]
            continue
        cat = classify(ep)
        agg[key][cat] += 1
        for flag in ("gold_in_issue", "gold_in_map", "gold_seen", "near_miss", "budget_bound"):
            diag_sum[key][flag] += d[flag]
        diag_sum[key]["invalid_frac"] += d["invalid_frac"]
        rows.append({"label": ep["label"], "toolset": ep["toolset"],
                     "instance_id": ep["instance_id"], "category": cat, **d,
                     "end": ep["end"], "manual_label": ""})

    print(f"{'cell':12s} {'n':>3s} {'hit':>4s} | {'no_ans':>6s} {'forced':>6s} {'wrong':>6s} | "
          f"{'seen%':>5s} {'near%':>5s} {'map%':>4s} {'budg%':>5s} {'inv':>4s}")
    result = {}
    for label in LABELS:
        for ts in TOOLSETS:
            key = (label, ts)
            a, ds = agg[key], diag_sum[key]
            fails = a["no_answer"] + a["forced_wrong"] + a["wrong_files"]
            if not a["n"]:
                continue
            f = max(fails, 1)
            row = {
                "n": a["n"], "hit": a["hit"], "no_answer": a["no_answer"],
                "forced_wrong": a["forced_wrong"], "wrong_files": a["wrong_files"],
                "fail_gold_seen": round(ds["gold_seen"] / f, 2),
                "fail_near_miss": round(ds["near_miss"] / f, 2),
                "fail_gold_in_map": round(ds["gold_in_map"] / f, 2),
                "fail_budget_bound": round(ds["budget_bound"] / f, 2),
                "fail_invalid_frac": round(ds["invalid_frac"] / f, 2),
            }
            result[f"{label}/{ts}"] = row
            print(f"{label+'/'+ts:12s} {a['n']:3d} {a['hit']:4d} | {a['no_answer']:6d} "
                  f"{a['forced_wrong']:6d} {a['wrong_files']:6d} | "
                  f"{row['fail_gold_seen']:5.2f} {row['fail_near_miss']:5.2f} "
                  f"{row['fail_gold_in_map']:4.2f} {row['fail_budget_bound']:5.2f} "
                  f"{row['fail_invalid_frac']:4.2f}")

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["label"], r["toolset"], r["instance_id"])))

    rng = random.Random(0)
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["label"], r["toolset"])].append(r)
    print(f"\n== stratified hand-label sample ({SAMPLE_PER_CELL}/cell) ==")
    for label in LABELS:
        for ts in TOOLSETS:
            pool = by_cell[(label, ts)]
            for r in rng.sample(pool, min(SAMPLE_PER_CELL, len(pool))):
                print(f"{label:9s} {ts}  {r['instance_id']:32s} {r['category']}")
    print(f"\nwrote {out_json} and {out_csv} ({len(rows)} failures)")


if __name__ == "__main__":
    main()
