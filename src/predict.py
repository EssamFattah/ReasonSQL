"""Ask a question about any SQLite database and run the generated query (safely, read-only).

  python -m src.predict --db data/spider_data/database/concert_singer/concert_singer.sqlite \
      --question "How many singers are older than 30?" --model artifacts/student_fused_q4
"""
import argparse

from src.common import (build_messages, chat_prompt, execute_sql, extract_sql, generate_text, get_schema,
                        load_model, sampling_for, split_reasoning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--show_reasoning", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=2048)
    args = ap.parse_args()

    model, tok = load_model(args.model, args.adapter_path)
    prompt = chat_prompt(tok, build_messages(get_schema(args.db), args.question), args.thinking)
    raw, st = generate_text(model, tok, prompt, args.max_tokens, **sampling_for(args.thinking))
    reasoning, answer, _ = split_reasoning(raw)
    sql = extract_sql(answer)
    if args.show_reasoning and reasoning:
        print("--- reasoning ---\n" + reasoning)
    print("--- SQL ---\n" + (sql or "(no SQL produced)"))
    ok, res = execute_sql(args.db, sql)
    print("--- result ---")
    if ok:
        for r in res["rows"][:20]:
            print(r)
        if len(res["rows"]) > 20:
            print(f"... {len(res['rows'])} rows total")
    else:
        print("error:", res)
    print(f"[{st['gen_tokens']} tokens, {st['total_s']:.1f}s, {st['gen_tps']:.0f} tok/s, "
          f"peak {st['peak_mem_gb']:.1f} GB]")


if __name__ == "__main__":
    main()
