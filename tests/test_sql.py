import pytest

from src.common import (check_sql, difficulty, error_category, execute_sql, extract_sql, first_statement,
                        item_seed, results_match, score_generation, split_reasoning, sql_features,
                        structural_diff)


# ---------------------------------------------------------------- extraction
@pytest.mark.parametrize("text,expected", [
    ("```sql\nSELECT 1;\n```", "SELECT 1"),
    ("```sqlite\nSELECT 2\n```", "SELECT 2"),
    ("```SQL SELECT 3```", "SELECT 3"),
    ("```\nSELECT 4\n```", "SELECT 4"),
    ("```SELECT 5```", "SELECT 5"),
    ("so the answer is SELECT 6 FROM t; done", "SELECT 6 FROM t"),
    ("```python\nx=1\n```\n```sql\nSELECT 7\n```", "SELECT 7"),
    ("```sql\nSELECT 1\n```\nfixed:\n```sql\nSELECT 8\n```", "SELECT 8"),  # last block wins
    ("no query here", ""),
])
def test_extract_sql(text, expected):
    assert extract_sql(text) == expected


def test_semicolons_inside_literals_and_comments():
    sql = "SELECT 'a;b', \"c;d\", [e;f] FROM t -- comment; here\nWHERE x = 'it''s;' /* ; */; DROP TABLE t"
    assert first_statement(sql) == sql[: sql.rindex(";")]


def test_whitespace_in_literals_preserved():
    assert extract_sql("```sql\nSELECT * FROM shop WHERE city = 'New  York'\n```").endswith("'New  York'")


def test_split_reasoning():
    assert split_reasoning("<think>\nplan\n</think>\n\nanswer") == ("plan", "answer", False)
    assert split_reasoning("plan only</think>answer") == ("plan only", "answer", False)
    assert split_reasoning("<think>never closes") == ("never closes", "", True)
    assert split_reasoning("no thinking") == ("", "no thinking", False)


# ---------------------------------------------------------------- execution & safety
def test_execute_select(db):
    ok, res = execute_sql(db, "SELECT count(*), avg(price) FROM item")
    assert ok and res["rows"] == [(10, 6.75)] and res["ncols"] == 2


@pytest.mark.parametrize("sql", [
    "ATTACH DATABASE '/etc/hosts' AS x",
    "PRAGMA table_info(item)",
    "DELETE FROM item",
    "CREATE TEMP TABLE z(a)",
])
def test_non_select_blocked(db, sql):
    ok, err = execute_sql(db, sql)
    assert not ok
    assert error_category(err) in {"blocked", "execution_error"}


def test_recursive_cte_allowed_and_timeout(db):
    ok, res = execute_sql(db, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x<5) "
                              "SELECT count(*) FROM c")
    assert ok and res["rows"] == [(5,)]
    ok, err = execute_sql(db, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c",
                          timeout=0.3)
    assert not ok and error_category(err) == "timeout"


def test_results_match():
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert not results_match([(1,), (1,)], [(1,)], ordered=False)  # multiset, not set
    assert results_match([(1.00001,)], [(1.0,)], ordered=True)
    assert results_match([(3,)], [(3.0,)], ordered=True)


@pytest.mark.parametrize("pred,category", [
    ("SELECT name FROM item WHERE price > 3 ORDER BY name", "correct"),
    ("SELECT name, price FROM item WHERE price > 3 ORDER BY name", "wrong_columns"),
    ("SELECT name FROM item WHERE price > 100", "empty_result"),
    ("SELECT name FROM item ORDER BY name", "wrong_row_count"),
    ("SELECT name FROM item WHERE price > 3 ORDER BY name DESC", "wrong_values"),
    ("SELECT nme FROM item", "missing_column"),
    ("SELECT name FROM items", "missing_table"),
    ("SELEC name FROM item", "syntax_error"),
    ("", "no_sql"),
])
def test_check_sql_categories(db, pred, category):
    res = check_sql(db, "SELECT name FROM item WHERE price > 3 ORDER BY name", pred)
    assert res["category"] == category
    assert res["correct"] == (category == "correct")


def test_gold_error_flagged(db):
    assert check_sql(db, "SELECT broken FROM nowhere", "SELECT 1")["category"] == "gold_error"


def test_score_generation_truncated(db):
    row = {"db_path": db, "gold_sql": "SELECT 1"}
    assert score_generation(row, "<think>still going", "length")["category"] == "truncated"
    ok = score_generation(row, "<think>x</think>```sql\nSELECT 1\n```", "stop")
    assert ok["correct"] and not ok["truncated"]


# ---------------------------------------------------------------- structure
def test_sql_features_and_difficulty():
    f = sql_features("SELECT T1.city, count(*) FROM shop T1 JOIN item T2 ON T1.id = T2.shop_id "
                     "WHERE T2.price > (SELECT avg(price) FROM item) GROUP BY T1.city HAVING count(*) > 1")
    assert f["joins"] == 1 and f["subqueries"] == 1 and f["group_by"] and f["having"] and f["tables"] == 2
    assert difficulty(f) in {"hard", "extra"}
    assert difficulty(sql_features("SELECT name FROM item")) == "easy"
    assert sql_features("this is not sql at all (((") is None
    assert difficulty(None) == "unparsed"


def test_structural_diff():
    gold = "SELECT a FROM t JOIN u ON t.i = u.i WHERE x > 1"
    assert structural_diff(gold, "SELECT a FROM t") == ["filtering", "joins"]
    assert structural_diff(gold, gold) == []


def test_item_seed_stable():
    assert item_seed(0, "dev_1") == item_seed(0, "dev_1")
    assert item_seed(0, "dev_1") != item_seed(1, "dev_1")


# ---------------------------------------------------------------- v3: ordering, test suite, export
@pytest.mark.parametrize("sql,ordered", [
    ("SELECT a FROM t ORDER BY a", True),
    ("SELECT a FROM (SELECT a FROM t ORDER BY a LIMIT 3)", False),
    ("SELECT a FROM t WHERE a IN (SELECT b FROM u ORDER BY b LIMIT 1)", False),
    ("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1", True),
    ("WITH x AS (SELECT a FROM t ORDER BY a) SELECT a FROM x", False),
])
def test_top_level_order(sql, ordered):
    from src.common import has_top_level_order
    assert has_top_level_order(sql) is ordered


def test_nested_order_by_compares_unordered(db):
    gold = "SELECT name FROM (SELECT name FROM item ORDER BY price DESC LIMIT 3)"
    pred = "SELECT name FROM item WHERE price >= 10.5"  # same 3 rows, different order
    assert check_sql(db, gold, pred)["correct"]


def test_test_suite_scoring(tmp_path, db):
    import shutil
    import sqlite3

    from src.evaluate_sql import export_official, test_suite_correct
    ts = tmp_path / "ts" / "shop"
    ts.mkdir(parents=True)
    shutil.copy(db, ts / "shop.sqlite")
    shutil.copy(db, ts / "shop_v2.sqlite")
    con = sqlite3.connect(ts / "shop_v2.sqlite")
    con.execute("INSERT INTO item VALUES (100, 'zz', 2.0, 1)")  # variant breaks a hard-coded shortcut
    con.commit()
    con.close()
    row = {"db_id": "shop", "gold_sql": "SELECT count(*) FROM item WHERE price < 3"}
    real = "SELECT count(*) FROM item WHERE price < 3"
    shortcut = "SELECT count(*) FROM item WHERE id < 2"  # correct on the original DB only
    assert check_sql(db, row["gold_sql"], shortcut)["correct"]
    assert test_suite_correct(tmp_path / "ts", row, real, True) is True
    assert test_suite_correct(tmp_path / "ts", row, shortcut, True) is False
    assert test_suite_correct(tmp_path / "ts", {**row, "db_id": "nope"}, real, True) is None
    export_official(tmp_path, [{"pred_sql": "SELECT\n 1 FROM shop WHERE city = 'New  York'", "gold_sql": "SELECT 1",
                                "db_id": "shop"},
                               {"pred_sql": "", "gold_sql": "SELECT 2", "db_id": "shop"}])
    assert (tmp_path / "official" / "pred.sql").read_text() == "SELECT 1 FROM shop WHERE city = 'New  York'\nSELECT 1\n"
    assert (tmp_path / "official" / "gold.sql").read_text().splitlines()[1] == "SELECT 2\tshop"


@pytest.mark.parametrize("sql,expected", [
    ("SELECT *\n  FROM shop\n WHERE city = 'New  York'", "SELECT * FROM shop WHERE city = 'New  York'"),
    ("SELECT \"col  name\" ,  [a  b] FROM t", "SELECT \"col  name\" , [a  b] FROM t"),
    ("SELECT 'it''s   ok' FROM t", "SELECT 'it''s   ok' FROM t"),
    ("SELECT a -- trailing; comment\nFROM t /* block\n comment */ WHERE b = 'x\ny'", "SELECT a FROM t WHERE b = 'x\ny'"),
    ("  SELECT 1  ", "SELECT 1"),
])
def test_sql_one_line_preserves_literals(sql, expected):
    from src.common import sql_one_line
    assert sql_one_line(sql) == expected


def test_sql_one_line_is_semantically_identical(db):
    from src.common import sql_one_line
    sql = "SELECT count(*)\nFROM shop -- note\nWHERE city = 'New  York'"
    assert execute_sql(db, sql) == execute_sql(db, sql_one_line(sql))
    assert execute_sql(db, sql)[1]["rows"] == [(1,)]
