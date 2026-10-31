"""Turn the raw Spider release into prompt-ready JSONL files.

Outputs:
  data/processed/train_pool.jsonl  questions the teacher will answer (shuffled)
  data/processed/dev.jsonl         held-out Spider dev set for evaluation (shuffled)
"""
import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from src.common import get_schema, write_jsonl


def load_split(spider_dir: Path, filename: str, prefix: str, sample_rows: int) -> list[dict]:
    raw = json.loads((spider_dir / filename).read_text())
    schema_cache: dict[str, str] = {}
    rows = []
    for i, ex in enumerate(tqdm(raw, desc=filename)):
        db_id = ex["db_id"]
        db_path = spider_dir / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            continue
        if db_id not in schema_cache:
            schema_cache[db_id] = get_schema(str(db_path), sample_rows)
        rows.append({
            "id": f"{prefix}_{i}",
            "db_id": db_id,
            "db_path": str(db_path.resolve()),
            "question": ex["question"].strip(),
            "gold_sql": ex["query"].strip(),
            "schema": schema_cache[db_id],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider_dir", default="data/spider_data")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--sample_rows", type=int, default=3, help="example rows per table in the prompt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spider = Path(args.spider_dir)
    out = Path(args.out_dir)
    rng = random.Random(args.seed)

    train = load_split(spider, "train_spider.json", "train", args.sample_rows)
    dev = load_split(spider, "dev.json", "dev", args.sample_rows)
    rng.shuffle(train)  # so --limit in later steps gives a random, DB-diverse subset
    rng.shuffle(dev)

    write_jsonl(out / "train_pool.jsonl", train)
    write_jsonl(out / "dev.jsonl", dev)
    print(f"train_pool: {len(train)} | dev: {len(dev)} | dbs: {len({r['db_id'] for r in train})} train, "
          f"{len({r['db_id'] for r in dev})} dev (disjoint by design in Spider)")


if __name__ == "__main__":
    main()
