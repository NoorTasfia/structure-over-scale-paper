"""Localization metrics against gold files/functions."""


def _norm_func(f):
    path, _, qual = f.partition("::")
    return f"{path.removeprefix('./')}::{qual}"


def score(pred_files, pred_funcs, gold_files, gold_funcs):
    gold_files = set(gold_files)
    gold_funcs = {_norm_func(f) for f in gold_funcs}
    pred_funcs = [_norm_func(f) for f in pred_funcs]
    top5 = set(pred_files[:5])
    s = {
        "file_acc1": bool(pred_files) and pred_files[0] in gold_files,
        "file_hit5": bool(top5 & gold_files),
        "file_all5": gold_files <= top5,
        "func_hit5": None,
    }
    if gold_funcs:
        s["func_hit5"] = bool(set(pred_funcs[:5]) & gold_funcs)
    return s
