"""Offline checks for the parts that don't need a model or network: python -m pytest tests"""
from gemma_nav import pyast
from gemma_nav.agent import parse_call
from gemma_nav.data import changed_lines
from gemma_nav.score import score
from gemma_nav.tools import Workspace

SRC = """\
class A:
    def f(self):
        x = 1
        return helper(x)

def helper(x):
    return x + 1
"""

PATCH = """\
diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -2,3 +2,3 @@ class A:
     def f(self):
-        x = 1
+        x = 2
         return helper(x)
@@ -6,2 +6,3 @@ def helper(x):
 def helper(x):
+    x = abs(x)
     return x + 1
diff --git a/pkg/new.py b/pkg/new.py
--- /dev/null
+++ b/pkg/new.py
@@ -0,0 +1 @@
+print("hi")
"""


def test_changed_lines_and_enclosing():
    lines = changed_lines(PATCH)
    assert dict(lines) == {"pkg/mod.py": {3, 6}}
    assert pyast.enclosing_functions(SRC, lines["pkg/mod.py"]) == {"A.f", "helper"}


def test_callees_and_outline():
    assert pyast.callees(SRC, "A.f") == ["helper"]
    assert pyast.outline(SRC)[1].strip().startswith("def f")


def test_parse_call():
    assert parse_call('think\n```json\n{"tool": "grep", "args": {"pattern": "x"}}\n```') == ("grep", {"pattern": "x"})
    assert parse_call("no json here") is None


def test_score():
    s = score(["a.py", "b.py"], ["./a.py::A.f"], ["b.py"], ["a.py::A.f"])
    assert s == {"file_acc1": False, "file_hit5": True, "file_all5": True, "func_hit5": True}


def test_workspace_tools(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(SRC)
    ws = Workspace(tmp_path)
    assert "mod.py" in ws.list_dir("pkg")
    assert "3:         x = 1" in ws.read_file("pkg/mod.py", 1, 5)
    assert "def helper" in ws.outline("pkg/mod.py")
    assert "pkg/mod.py" in ws.repo_map("helper is broken in mod")
    assert "pkg/mod.py:6" in ws.find_definition("helper")


def test_api_view_elides_old_observations():
    from gemma_nav.agent import api_view
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "issue"}]
    for i in range(6):
        msgs.append({"role": "assistant", "content": f"a{i}"})
        msgs.append({"role": "user", "content": f"Observation:\nobs{i}"})
    v = api_view(msgs)
    assert len(v) == len(msgs)
    assert v[1]["content"] == "issue"                      # issue kept
    assert all(f"a{i}" in v[2 + 2 * i]["content"] for i in range(6))  # assistant turns kept
    assert "elided" in v[3]["content"] and "obs0" not in v[3]["content"]
    assert v[-1]["content"].endswith("obs5")               # last 3 observations kept
    assert v[-3]["content"].endswith("obs4")
    assert v[-5]["content"].endswith("obs3")
    assert "elided" in v[-7]["content"]


def test_parse_call_gemma_native_syntax():
    # exact shapes observed in E4B-Q4 trajectories
    assert parse_call('<|tool_call>call:list_dir(path=".")\n') == ("list_dir", {"path": "."})
    assert parse_call('thinking first.<|tool_call>call:grep(r"style|library", path="src/")') == \
        ("grep", {"pattern": "style|library", "path": "src/"})
    assert parse_call('<|tool_call|>call: read_file("a/b.py", 10, 60)') == \
        ("read_file", {"path": "a/b.py", "start": 10, "end": 60})
    assert parse_call('<|tool_call>call:grep(re.compile("x"))') is None  # non-literal arg


def test_parse_call_flattened_and_fused_json():
    # exact shapes observed in E2B-Q8/BF16 trajectories: args hoisted to the top level...
    txt = '```json\n{"tool": "submit", "files": ["a.py"], "functions": ["a.py::A.f"]}\n```'
    assert parse_call(txt) == ("submit", {"files": ["a.py"], "functions": ["a.py::A.f"]})
    # ...and a Python-style call fused into the tool field (common in E2B-Q4 too)
    txt2 = '```json\n{"tool": "grep(pattern=\\"foo\\", path=\\"src\\")"}\n```'
    assert parse_call(txt2) == ("grep", {"pattern": "foo", "path": "src"})
    assert parse_call('```json\n{"tool": "list_dir(path=\\"\\")"}\n```') == ("list_dir", {"path": ""})
    # fused call with non-literal args stays invalid
    assert parse_call('```json\n{"tool": "grep(re.compile(1))"}\n```') is None


def test_parse_call_salvages_unbalanced_json():
    txt = '```json\n{"tool": "submit", "args": {"files": ["a.py"], "functions": ["A.f"]}\n```'
    assert parse_call(txt) == ("submit", {"files": ["a.py"], "functions": ["A.f"]})
    # fence never closed (reply truncated at token limit)
    txt2 = 'ok\n```json\n{"tool": "outline", "args": {"path": "x.py"}}'
    assert parse_call(txt2) == ("outline", {"path": "x.py"})
