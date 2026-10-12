"""Handcrafted structural features for Python source code (ast + text stats)."""
import ast
import io
import re
import tokenize

import numpy as np

AST_NAMES = ["parse_ok", "n_nodes", "max_depth", "n_funcs", "n_classes", "avg_func_len",
             "r_if", "r_for", "r_while", "r_try", "r_comp"]
TEXT_NAMES = ["n_lines", "avg_line_len", "comment_ratio", "has_docstring", "blank_ratio",
              "avg_ident_len", "snake_frac", "ident_unique_ratio"]
FEATURE_NAMES = AST_NAMES + TEXT_NAMES
COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
SNAKE = re.compile(r"^[a-z_][a-z0-9_]*$")
IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _depth(node, d=0):
    return max([d] + [_depth(c, d + 1) for c in ast.iter_child_nodes(node)])


def _ast_feats(src):
    try:
        tree = ast.parse(src)
        max_depth = _depth(tree)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return [0.0] * len(AST_NAMES), None
    nodes = list(ast.walk(tree))
    n = len(nodes)
    funcs = [x for x in nodes if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]
    func_lens = [x.end_lineno - x.lineno + 1 for x in funcs]
    count = lambda t: sum(isinstance(x, t) for x in nodes) / n
    has_doc = any(ast.get_docstring(x, clean=False) for x in nodes
                  if isinstance(x, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    feats = [1.0, n, max_depth, len(funcs), sum(isinstance(x, ast.ClassDef) for x in nodes),
             float(np.mean(func_lens)) if func_lens else 0.0,
             count(ast.If), count((ast.For, ast.AsyncFor)), count(ast.While), count(ast.Try),
             count(COMPS)]
    return feats, (has_doc, nodes)


def _comment_lines(src):
    try:
        return {t.start[0] for t in tokenize.generate_tokens(io.StringIO(src).readline)
                if t.type == tokenize.COMMENT}
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return {i for i, l in enumerate(src.splitlines(), 1) if l.lstrip().startswith("#")}


def extract(src):
    ast_f, parsed = _ast_feats(src)
    lines = src.splitlines() or [""]
    n_lines = len(lines)
    if parsed:
        has_doc = float(parsed[0])
        idents = [x.id for x in parsed[1] if isinstance(x, ast.Name)]
        idents += [x.name for x in parsed[1] if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                 ast.ClassDef))]
        idents += [x.arg for x in parsed[1] if isinstance(x, ast.arg)]
    else:
        has_doc = float('"""' in src or "'''" in src)
        idents = IDENT.findall(src)
    text_f = [n_lines, float(np.mean([len(l) for l in lines])),
              len(_comment_lines(src)) / n_lines, has_doc,
              sum(not l.strip() for l in lines) / n_lines,
              float(np.mean([len(i) for i in idents])) if idents else 0.0,
              sum(bool(SNAKE.match(i)) for i in idents) / len(idents) if idents else 0.0,
              len(set(idents)) / len(idents) if idents else 0.0]
    return ast_f + text_f


def extract_all(codes):
    return np.array([extract(c) for c in codes], dtype=np.float32)


def fit_stats(x):
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1.0
    return mean, std


def standardize(x, mean, std):
    return ((x - mean) / std).astype(np.float32)
