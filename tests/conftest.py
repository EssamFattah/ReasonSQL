import sqlite3

import pytest

try:
    import sqlglot  # noqa: F401
except ImportError as e:  # make a missing dependency obvious instead of failing 2 random tests
    raise RuntimeError("sqlglot is required (SQL parsing/analysis): pip install -r requirements.txt") from e

from src.common import get_schema, write_jsonl


@pytest.fixture(autouse=True)
def offline_hf(monkeypatch):
    """Tests never touch the network: pretend every HF repo resolves to a fixed revision."""
    from src import fingerprint
    def fake_snapshot(model):
        repo, rev = fingerprint.split_revision(model)
        commit = rev or "test-revision"
        return f"/fake/hf/{repo}/snapshots/{commit}", commit
    monkeypatch.setattr(fingerprint, "resolve_hf_snapshot", fake_snapshot)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE item(id INTEGER PRIMARY KEY, name TEXT, price REAL, shop_id INT);
        CREATE TABLE shop(id INTEGER PRIMARY KEY, city TEXT);
        INSERT INTO shop VALUES (1, 'New  York'), (2, 'Paris');
    """)
    con.executemany("INSERT INTO item VALUES (?,?,?,?)", [(i, f"n{i}", i * 1.5, 1 + i % 2) for i in range(10)])
    con.commit()
    con.close()
    return str(path)


@pytest.fixture
def pool_and_traces(tmp_path, db):
    """30 questions. Teacher pattern: q%3==0 solved at attempt 0, q%3==1 at attempt 1, q%3==2 never."""
    schema = get_schema(db)
    pool, traces = [], []
    for i in range(30):
        gold = f"SELECT name FROM item WHERE price > {i % 10} ORDER BY name"
        pool.append({"id": f"train_{i:02d}", "db_id": "shop", "db_path": db, "question": f"q{i}",
                     "gold_sql": gold, "schema": schema})
        solved_at = {0: 0, 1: 1, 2: None}[i % 3]
        for k in range(3):
            ok = solved_at == k
            traces.append({"id": f"train_{i:02d}", "sample_idx": k, "reasoning": f"r{i}-{k}",
                           "pred_sql": gold if ok else "SELECT 1", "truncated": False, "correct": ok,
                           "category": "correct" if ok else "wrong_columns", "n_tokens": 100, "gen_time_s": 2.0,
                           "example_tokens": 500})
            if ok:
                break
    write_jsonl(tmp_path / "pool.jsonl", pool)
    write_jsonl(tmp_path / "traces.jsonl", traces)
    return tmp_path / "pool.jsonl", tmp_path / "traces.jsonl", traces
