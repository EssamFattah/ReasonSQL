"""Shared utilities: prompts, schema extraction, SQL parsing/analysis, safe execution, scoring, MLX helpers."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a SQLite database schema and a question, "
    "write a single SQLite query that answers the question. "
    "Return the final query inside a ```sql code block."
)

# Evaluation protocol (fixed for every model so comparisons *within a mode* are apples-to-apples):
#   thinking mode -> Qwen3's recommended sampling (greedy decoding causes repetition loops when thinking)
#   no-thinking   -> greedy
# Every question gets its own deterministic seed, so results don't depend on run order or resumption.
THINK_SAMPLING = dict(temp=0.6, top_p=0.95, top_k=20)
NOTHINK_SAMPLING = dict(temp=0.0, top_p=1.0, top_k=0)


# A training example must fit the student's context window *including the prompt*; mlx_lm would
# otherwise truncate it and cut off the SQL target. Keep equal to train.py --max_seq_length.
DEFAULT_MAX_EXAMPLE_TOKENS = 4096


def format_target(reasoning: str, sql: str) -> str:
    return f"<think>\n{reasoning}\n</think>\n\n```sql\n{sql}\n```"


def example_token_count(tokenizer, messages: list[dict], reasoning: str, sql: str) -> int:
    """Token length of the full rendered training example (reasoning format = the longest variant)."""
    msgs = messages + [{"role": "assistant", "content": format_target(reasoning, sql)}]
    return len(tokenizer.apply_chat_template(msgs, tokenize=True))


def is_well_formed(t: dict, max_example_tokens: int = DEFAULT_MAX_EXAMPLE_TOKENS, max_chars: int | None = None) -> bool:
    """Passes the cheap format filters any pipeline would apply (no correctness check)."""
    if not t["pred_sql"] or t["truncated"]:
        return False
    if max_chars is not None and len(t["reasoning"]) > max_chars:
        return False
    n = t.get("example_tokens")
    if n is None:
        raise KeyError(f"trace {t['id']}#{t['sample_idx']} has no example_tokens; "
                       "regenerate or run build_sft with --fill_tokens")
    return n <= max_example_tokens


def is_usable(t: dict, max_example_tokens: int = DEFAULT_MAX_EXAMPLE_TOKENS, max_chars: int | None = None) -> bool:
    """THE definition of a usable verified trace. Used by generation (when to stop sampling and
    what to skip on resume), teacher_stats (yield) and build_sft (what can be trained on)."""
    return bool(t["correct"]) and is_well_formed(t, max_example_tokens, max_chars)


def sampling_for(thinking: bool) -> dict:
    return dict(THINK_SAMPLING if thinking else NOTHINK_SAMPLING)


def item_seed(base_seed: int, *parts) -> int:
    """Stable per-item seed (Python's hash() is randomised per process, so don't use it)."""
    h = base_seed * 1_000_003
    for p in parts:
        for ch in str(p):
            h = (h * 31 + ord(ch)) % (2**31 - 1)
    return h


# ---------------------------------------------------------------- io
def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, p)


def append_jsonl(path, row) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def write_json(path, obj) -> None:
    """Atomic write: a result file either exists complete or not at all (used as a completion marker)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, p)


# ---------------------------------------------------------------- schema / prompts
def _connect_ro(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="ignore")  # some Spider DBs contain bad utf-8
    return con


def get_schema(db_path: str, sample_rows: int = 3) -> str:
    """CREATE TABLE statements (+ optional example rows) for every table in the DB."""
    con = _connect_ro(db_path)
    tables = con.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    parts = []
    for name, sql in tables:
        parts.append(sql.strip().rstrip(";") + ";")
        if sample_rows > 0:
            try:
                cur = con.execute(f'SELECT * FROM "{name}" LIMIT {sample_rows}')
                cols = [d[0] for d in cur.description]
                vals = cur.fetchall()
                if vals:
                    body = "\n".join("\t".join(str(v)[:50] for v in r) for r in vals)
                    parts.append(f"/* {len(vals)} example rows from {name}:\n" + "\t".join(cols) + "\n" + body + "\n*/")
            except sqlite3.Error:
                pass
    con.close()
    return "\n\n".join(parts)


def build_messages(schema: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Database schema:\n{schema}\n\nQuestion: {question}"},
    ]


# ---------------------------------------------------------------- parsing model output
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_SQL_BLOCK_RE = re.compile(r"```(?:sqlite|sql)\b[ \t]*\n?(.*?)```", re.S | re.I)
_ANY_BLOCK_RE = re.compile(r"```[A-Za-z]*[ \t]*\n(.*?)```|```(.*?)```", re.S)


def split_reasoning(text: str) -> tuple[str, str, bool]:
    """Return (reasoning, answer, unclosed). unclosed=True if the thinking block never ended."""
    m = _THINK_RE.search(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip(), False
    if "</think>" in text:
        r, a = text.split("</think>", 1)
        return r.replace("<think>", "").strip(), a.strip(), False
    if "<think>" in text:
        return text.replace("<think>", "").strip(), "", True
    return "", text.strip(), False


def first_statement(sql: str) -> str:
    """Cut at the first ';' that is outside string literals, quoted identifiers and comments."""
    quote = None
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if quote:
            if c == quote:
                if i + 1 < n and sql[i + 1] == quote:  # escaped '' or ""
                    i += 2
                    continue
                quote = None
        elif c in "'\"`":
            quote = c
        elif c == "[":
            quote = "]"
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            if j == -1:
                return sql[:i]
            i = j
            continue
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            if j == -1:
                return sql[:i]
            i = j + 2
            continue
        elif c == ";":
            return sql[:i]
        i += 1
    return sql


def sql_one_line(sql: str) -> str:
    """Put a query on one line WITHOUT touching string literals or quoted identifiers.

    Whitespace outside quotes collapses to single spaces. Comments are dropped, because a
    `-- comment` on a single line would swallow the rest of the query.
    """
    out, quote, i, n = [], None, 0, len(sql)
    pending_space = False
    while i < n:
        c = sql[i]
        if quote:
            out.append(c)
            if c == quote:
                if i + 1 < n and sql[i + 1] == quote:  # escaped '' / ""
                    out.append(sql[i + 1])
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j == -1 else j
            pending_space = True
            continue
        if sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            i = n if j == -1 else j + 2
            pending_space = True
            continue
        if c.isspace():
            pending_space = True
            i += 1
            continue
        if pending_space and out:
            out.append(" ")
        pending_space = False
        if c in "'\"`":
            quote = c
        elif c == "[":
            quote = "]"
        out.append(c)
        i += 1
    return "".join(out)


def extract_sql(answer: str) -> str:
    """Last ```sql block, else last ``` block, else bare SELECT/WITH. Whitespace inside literals is preserved."""
    blocks = _SQL_BLOCK_RE.findall(answer) or [a or b for a, b in _ANY_BLOCK_RE.findall(answer)]
    if blocks:
        sql = blocks[-1]
    else:
        m = re.search(r"\b(SELECT|WITH)\b.*", answer, re.S | re.I)
        sql = m.group(0) if m else ""
    return first_statement(sql.strip()).strip()


# ---------------------------------------------------------------- SQL structure analysis (sqlglot)
def sql_features(sql: str) -> dict | None:
    try:
        import sqlglot
        from sqlglot import exp

        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return None
    if tree is None:
        return None

    def select_depth(node) -> int:
        d, p = 0, node.parent
        while p is not None:
            d += isinstance(p, exp.Select)
            p = p.parent
        return d

    depths = [select_depth(s) for s in tree.find_all(exp.Select)]
    return {
        "joins": len(list(tree.find_all(exp.Join))),
        "tables": len({t.name.lower() for t in tree.find_all(exp.Table) if t.name}),
        "subqueries": sum(d > 0 for d in depths),
        "nesting_depth": max(depths, default=0),
        "set_ops": len(list(tree.find_all(exp.Union, exp.Intersect, exp.Except))),
        "aggregations": len(list(tree.find_all(exp.AggFunc))),
        "group_by": tree.find(exp.Group) is not None,
        "having": tree.find(exp.Having) is not None,
        "where": tree.find(exp.Where) is not None,
        "order_by": tree.find(exp.Order) is not None,
        "limit": tree.find(exp.Limit) is not None,
        "distinct": tree.find(exp.Distinct) is not None,
    }


def has_top_level_order(sql: str) -> bool:
    """True only if the OUTERMOST query is ordered (an ORDER BY inside a subquery doesn't count)."""
    try:
        import sqlglot

        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        tree = None
    if tree is None:  # unparseable: conservative regex fallback
        return bool(re.search(r"\border\s+by\b", sql, re.I))
    return tree.args.get("order") is not None


def difficulty(f: dict | None) -> str:
    """Spider-style hardness proxy computed from the gold query's structure."""
    if f is None:
        return "unparsed"
    score = (f["joins"] + 2 * f["subqueries"] + 2 * f["set_ops"] + f["group_by"] + f["having"]
             + (f["aggregations"] > 0) + f["order_by"])
    return "easy" if score <= 1 else "medium" if score <= 3 else "hard" if score <= 5 else "extra"


_STRUCT_CHECKS = [  # (feature, label, compare as count?)
    ("joins", "joins", True), ("subqueries", "nesting", True), ("set_ops", "set_ops", True),
    ("aggregations", "aggregation", False), ("where", "filtering", False), ("group_by", "grouping", False),
    ("having", "grouping", False), ("order_by", "ordering", False), ("limit", "ordering", False),
    ("distinct", "distinct", False),
]


def structural_diff(gold_sql: str, pred_sql: str) -> list[str]:
    """Which structural aspects differ between gold and a wrong prediction (heuristic error analysis)."""
    fg, fp = sql_features(gold_sql), sql_features(pred_sql)
    if fg is None or fp is None:
        return []
    out = set()
    for key, label, as_count in _STRUCT_CHECKS:
        a, b = fg[key], fp[key]
        if (a != b) if as_count else (bool(a) != bool(b)):
            out.add(label)
    return sorted(out)


# ---------------------------------------------------------------- safe execution
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                    getattr(sqlite3, "SQLITE_RECURSIVE", 33)}


def _authorizer(action, *_):
    # Read-only mode already blocks writes; this also blocks ATTACH, PRAGMA, temp tables, etc.
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def execute_sql(db_path: str, sql: str, timeout: float = 10.0):
    """Returns (True, {"rows", "ncols"}) or (False, error_message)."""
    if not sql:
        return False, "empty query"
    con = _connect_ro(db_path)
    con.set_authorizer(_authorizer)
    start = time.monotonic()
    con.set_progress_handler(lambda: 1 if time.monotonic() - start > timeout else 0, 10_000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchall()
        return True, {"rows": rows, "ncols": len(cur.description or [])}
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    finally:
        con.close()


def error_category(err: str) -> str:
    e = err.lower()
    if e == "empty query":
        return "no_sql"
    if "interrupted" in e:
        return "timeout"
    if "not authorized" in e or "prohibited" in e:
        return "blocked"
    if "no such table" in e:
        return "missing_table"
    if "no such column" in e or "ambiguous column" in e:
        return "missing_column"
    if "syntax error" in e or "incomplete input" in e or "unrecognized token" in e:
        return "syntax_error"
    return "execution_error"


def _norm(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, str):
        return v.strip()
    return v


def results_match(gold_rows, pred_rows, ordered: bool) -> bool:
    g = [tuple(_norm(v) for v in r) for r in gold_rows]
    p = [tuple(_norm(v) for v in r) for r in pred_rows]
    return g == p if ordered else Counter(g) == Counter(p)


def check_sql(db_path: str, gold_sql: str, pred_sql: str, timeout: float = 10.0) -> dict:
    """Execution-based verification + a failure category for anything that isn't correct."""
    ok_g, g = execute_sql(db_path, gold_sql, timeout)
    if not ok_g:
        return {"valid": False, "correct": False, "error": f"GOLD_ERROR {g}", "category": "gold_error",
                "gold_empty": None}
    base = {"gold_empty": len(g["rows"]) == 0, "gold_rows": len(g["rows"])}
    ok_p, p = execute_sql(db_path, pred_sql, timeout)
    if not ok_p:
        return {**base, "valid": False, "correct": False, "error": p, "category": error_category(p)}
    correct = results_match(g["rows"], p["rows"], has_top_level_order(gold_sql))
    if correct:
        cat = "correct"
    elif p["ncols"] != g["ncols"]:
        cat = "wrong_columns"
    elif not p["rows"]:
        cat = "empty_result"
    elif len(p["rows"]) != len(g["rows"]):
        cat = "wrong_row_count"
    else:
        cat = "wrong_values"
    return {**base, "valid": True, "correct": correct, "error": None, "category": cat, "pred_rows": len(p["rows"])}


def score_generation(row: dict, raw: str, finish_reason: str | None = None) -> dict:
    """Parse a raw generation and verify it against the gold query on the real DB."""
    reasoning, answer, unclosed = split_reasoning(raw)
    pred = extract_sql(answer)
    res = check_sql(row["db_path"], row["gold_sql"], pred)
    if not pred and (unclosed or finish_reason == "length"):
        res["category"] = "truncated"
    if res["valid"] and not res["correct"]:
        res["struct_diff"] = structural_diff(row["gold_sql"], pred)
    return {"reasoning": reasoning, "answer": answer, "pred_sql": pred,
            "truncated": unclosed or finish_reason == "length", **res}


def tally(values) -> dict:
    return dict(Counter(values).most_common())


def group_by(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return out


# ---------------------------------------------------------------- MLX helpers (lazy imports)
def load_model(model: str, adapter_path: str | None = None):
    """Load from the exact pinned snapshot recorded in the run's spec (never a fresh lookup by repo name)."""
    from mlx_lm import load

    from src.fingerprint import resolve_model_path

    return load(resolve_model_path(model), adapter_path=adapter_path)


def chat_prompt(tokenizer, messages: list[dict], enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


def set_seed(seed: int) -> None:
    import mlx.core as mx

    mx.random.seed(seed)


def generate_text(model, tokenizer, prompt: str, max_tokens: int, temp=0.0, top_p=1.0, top_k=0):
    """Stream a generation. Returns (text, stats) with time-to-first-token, throughput and peak memory."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
    t0 = time.perf_counter()
    ttft, parts, last = None, [], None
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
        if ttft is None:
            ttft = time.perf_counter() - t0
        parts.append(resp.text)
        last = resp
    stats = {
        "ttft_s": ttft, "total_s": time.perf_counter() - t0,
        "prompt_tokens": getattr(last, "prompt_tokens", None),
        "gen_tokens": getattr(last, "generation_tokens", None),
        "prompt_tps": getattr(last, "prompt_tps", None),
        "gen_tps": getattr(last, "generation_tps", None),
        "peak_mem_gb": getattr(last, "peak_memory", None),
        "finish_reason": getattr(last, "finish_reason", None),
    }
    return "".join(parts), stats


def reset_peak_memory() -> None:
    import mlx.core as mx

    fn = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
    fn()


def peak_memory_gb() -> float:
    import mlx.core as mx

    fn = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return fn() / 1e9


def n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
