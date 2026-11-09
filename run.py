"""Run one (model, precision, toolset) cell over a sample of SWE-bench Lite. Resumable.

Example:
  python run.py --base-url http://localhost:8000/v1 --model gemma --label small-q4 --toolset B --n 60
"""
import argparse
import json
from pathlib import Path

import openai

from gemma_nav.agent import run_episode
from gemma_nav.data import checkout, gold_labels, load_tasks
from gemma_nav.score import score
from gemma_nav.tools import Workspace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--label", required=True, help="model+precision tag, e.g. small-q4")
    ap.add_argument("--toolset", choices=["A", "B", "C"], required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--cache", default="cache/repos")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    out = Path(args.out) / f"{args.label}__{args.toolset}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["instance_id"] for line in out.open(encoding="utf-8")}

    client = openai.OpenAI(base_url=args.base_url, api_key="none", timeout=600)
    tasks = sorted(load_tasks(args.n, args.seed), key=lambda t: (t["repo"], t["instance_id"]))
    for i, task in enumerate(tasks, 1):
        if task["instance_id"] in done:
            continue
        repo_dir = checkout(task, args.cache)
        gold_files, gold_funcs = gold_labels(task, repo_dir)
        ep = run_episode(client, args.model, Workspace(repo_dir), task["problem_statement"],
                         args.toolset, args.max_turns)
        rec = {
            "instance_id": task["instance_id"], "repo": task["repo"],
            "label": args.label, "toolset": args.toolset,
            "gold_files": gold_files, "gold_functions": gold_funcs,
            **score(ep["files"], ep["functions"], gold_files, gold_funcs),
            **ep,
        }
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i}/{len(tasks)}] {task['instance_id']} end={ep['end']} "
              f"file@1={rec['file_acc1']} file@5={rec['file_hit5']} func@5={rec['func_hit5']} {ep['seconds']}s")


if __name__ == "__main__":
    main()
