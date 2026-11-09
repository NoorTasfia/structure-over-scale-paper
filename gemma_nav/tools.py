"""The three tool conditions.

A: list_dir / read_file / grep (baseline)
B: A + outline / find_definition / find_references / callees (code graph)
C: A + a static, issue-ranked repo map in the system prompt
"""
import itertools
import re
import subprocess
from pathlib import Path

from . import pyast

MAX_READ_LINES = 200
MAX_GREP_LINES = 50
MAX_GRAPH_RESULTS = 30


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self._jedi_project = None

    def _path(self, rel):
        p = (self.root / rel).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("path escapes the repository")
        return p

    def _rel(self, p):
        return Path(p).resolve().relative_to(self.root).as_posix()

    # --- Toolset A ---------------------------------------------------------

    def list_dir(self, path="."):
        p = self._path(path)
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
        return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries if e.name != ".git")

    def read_file(self, path, start=1, end=None):
        lines = self._path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(int(start), 1)
        end = min(int(end) if end else start + MAX_READ_LINES - 1, start + MAX_READ_LINES - 1, len(lines))
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        return f"{path} lines {start}-{end} of {len(lines)}\n{body}"

    def grep(self, pattern, path="."):
        out = subprocess.run(
            ["git", "-C", str(self.root), "grep", "-n", "-I", "-E", pattern, "--", path],
            capture_output=True, encoding="utf-8", errors="replace",  # not text=True: that is cp1252 on Windows
        )
        if out.returncode == 1:
            return "no matches"
        if out.returncode != 0:
            raise ValueError(out.stderr.strip())
        hits = out.stdout.splitlines()
        more = f"\n... {len(hits) - MAX_GREP_LINES} more matches" if len(hits) > MAX_GREP_LINES else ""
        return "\n".join(h[:200] for h in hits[:MAX_GREP_LINES]) + more

    # --- Toolset B ---------------------------------------------------------

    def _project(self):
        if self._jedi_project is None:
            import jedi
            self._jedi_project = jedi.Project(self.root)
        return self._jedi_project

    def _search(self, name):
        last = name.rsplit(".", 1)[-1]
        found = []
        for n in self._project().search(last, all_scopes=True):
            if n.module_path is None or not Path(n.module_path).resolve().is_relative_to(self.root):
                continue
            if "." in name and not (n.full_name or "").endswith(name):
                continue
            found.append(n)
            if len(found) >= MAX_GRAPH_RESULTS:
                break
        return found

    def outline(self, path):
        rows = pyast.outline(self._path(path).read_text(encoding="utf-8", errors="replace"))
        return "\n".join(rows) or "no classes or functions"

    def find_definition(self, name):
        found = self._search(name)
        return "\n".join(
            f"{self._rel(n.module_path)}:{n.line}  {n.type} {n.full_name}" for n in found
        ) or "not found"

    def find_references(self, name):
        import jedi

        found = self._search(name)
        if not found:
            return "definition not found"
        d = found[0]
        script = jedi.Script(path=d.module_path, project=self._project())
        refs = itertools.islice(
            (r for r in script.get_references(d.line, d.column, scope="project")
             if r.module_path and Path(r.module_path).resolve().is_relative_to(self.root)),
            MAX_GRAPH_RESULTS,
        )
        rows = [f"{self._rel(r.module_path)}:{r.line}  {(r.get_line_code() or '').strip()[:150]}" for r in refs]
        return f"references to {d.full_name}:\n" + "\n".join(rows)

    def callees(self, path, function):
        source = self._path(path).read_text(encoding="utf-8", errors="replace")
        return "\n".join(pyast.callees(source, function)) or "no calls"

    # --- Toolset C ---------------------------------------------------------

    def repo_map(self, issue, budget_chars=12000):
        """Outlines of the non-test files whose paths/definitions best overlap the issue text."""
        terms = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", issue)}
        scored = []
        for p in self.root.rglob("*.py"):
            rel = self._rel(p)
            if ".git/" in rel or "test" in rel.split("/")[0] or "/tests/" in rel:
                continue
            try:
                rows = pyast.outline(p.read_text(encoding="utf-8", errors="replace"))
            except (SyntaxError, ValueError):
                continue
            words = set(re.findall(r"[a-z_][a-z0-9_]{2,}", (rel + " " + " ".join(rows)).lower()))
            scored.append((len(terms & words), rel, rows))
        scored.sort(key=lambda s: (-s[0], s[1]))
        parts, used = [], 0
        for score, rel, rows in scored:
            block = rel + "\n" + "\n".join("  " + r for r in rows) + "\n"
            if score == 0 or used + len(block) > budget_chars:
                break
            parts.append(block)
            used += len(block)
        return "".join(parts)


TOOL_DOCS = {
    "list_dir": 'list_dir(path=".") - list a directory',
    "read_file": "read_file(path, start=1, end=None) - read up to 200 numbered lines",
    "grep": 'grep(pattern, path=".") - extended-regex search over tracked files',
    "outline": "outline(path) - classes/functions in a Python file with line ranges",
    "find_definition": "find_definition(name) - where a function/class/method is defined (name or Class.method)",
    "find_references": "find_references(name) - call sites and other uses of a definition",
    "callees": "callees(path, function) - what a function calls (function as Qual.name)",
}

TOOLSETS = {
    "A": ["list_dir", "read_file", "grep"],
    "B": ["list_dir", "read_file", "grep", "outline", "find_definition", "find_references", "callees"],
    "C": ["list_dir", "read_file", "grep"],
}
