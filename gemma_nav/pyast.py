"""Small ast helpers shared by the graph tools, the repo map and gold-label extraction."""
import ast


def definitions(source):
    """Yield (qualname, node) for every class/function, including nested ones."""
    tree = ast.parse(source)

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = prefix + child.name
                yield qual, child
                yield from walk(child, qual + ".")

    yield from walk(tree, "")


def enclosing_functions(source, lines):
    """Qualnames of the innermost functions containing any of the given line numbers."""
    funcs = [
        (qual, node.lineno, node.end_lineno)
        for qual, node in definitions(source)
        if not isinstance(node, ast.ClassDef)
    ]
    found = set()
    for line in lines:
        inner = [f for f in funcs if f[1] <= line <= f[2]]
        if inner:
            found.add(max(inner, key=lambda f: f[1])[0])
    return found


def outline(source):
    """One line per definition: indented qualname with its line range."""
    rows = []
    for qual, node in definitions(source):
        kind = "class" if isinstance(node, ast.ClassDef) else "def"
        depth = qual.count(".")
        rows.append(f"{'  ' * depth}{kind} {node.name}  [L{node.lineno}-{node.end_lineno}]")
    return rows


def callees(source, qualname):
    """Names called inside the given function, in first-seen order."""
    for qual, node in definitions(source):
        if qual == qualname:
            seen = []
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    name = ast.unparse(sub.func)
                    if name not in seen:
                        seen.append(name)
            return seen
    raise ValueError(f"function {qualname!r} not found")
