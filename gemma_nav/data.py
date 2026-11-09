"""SWE-bench Lite tasks, repo checkout, and gold localization labels from the reference patch."""
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from .pyast import enclosing_functions

HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def load_tasks(n, seed=0):
    from datasets import load_dataset

    rows = list(load_dataset("princeton-nlp/SWE-bench_Lite", split="test"))
    rows.sort(key=lambda r: r["instance_id"])
    return random.Random(seed).sample(rows, n)


def _git(*args):
    out = subprocess.run(["git", *map(str, args)], capture_output=True,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(map(str, args))} failed: {out.stderr.strip()[:500]}")


def checkout(task, cache_dir):
    repo_dir = Path(cache_dir) / task["repo"].replace("/", "__")
    if not repo_dir.exists():
        _git("clone", "--quiet", f"https://github.com/{task['repo']}.git", repo_dir)
    _git("-C", repo_dir, "checkout", "--quiet", "-f", task["base_commit"])
    _git("-C", repo_dir, "clean", "-fdxq")
    return repo_dir


def changed_lines(patch):
    """Map each pre-existing file in a unified diff to the old-file line numbers it touches."""
    lines = defaultdict(set)
    path, cur = None, None
    for line in patch.splitlines():
        if line.startswith("--- "):
            path = line[6:] if line.startswith("--- a/") else None
            cur = None
        elif line.startswith("+++ ") or line.startswith("diff --git"):
            cur = None
        elif m := HUNK.match(line):
            cur = int(m.group(1))
        elif path is None or cur is None or line.startswith("\\"):
            continue
        elif line.startswith("-"):
            lines[path].add(cur)
            cur += 1
        elif line.startswith("+"):
            # Insertions are attributed to the preceding old line.
            lines[path].add(max(cur - 1, 1))
        else:
            cur += 1
    return lines


def gold_labels(task, repo_dir):
    """Gold files and 'path::Qual.name' functions edited by the reference patch."""
    touched = changed_lines(task["patch"])
    files = sorted(touched)
    funcs = set()
    for path, lines in touched.items():
        if path.endswith(".py"):
            source = (Path(repo_dir) / path).read_text(encoding="utf-8", errors="replace")
            funcs |= {f"{path}::{q}" for q in enclosing_functions(source, lines)}
    return files, sorted(funcs)
