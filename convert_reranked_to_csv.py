import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_concatenated_json(raw_text: str) -> List[Dict[str, Any]]:
    """Parse multiple JSON objects concatenated in one file."""
    decoder = json.JSONDecoder()
    items: List[Dict[str, Any]] = []
    idx = 0
    length = len(raw_text)

    while idx < length:
        while idx < length and raw_text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, next_idx = decoder.raw_decode(raw_text, idx)
        if isinstance(obj, dict):
            items.append(obj)
        idx = next_idx

    return items


def to_csv_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for block_index, record in enumerate(records, start=1):
        base = {
            "block_index": block_index,
            "query": record.get("query"),
            "source": record.get("source"),
            "candidate_length": record.get("candidate_length"),
            "retrieved_length": record.get("length"),
        }
        results = record.get("results") or []
        if not isinstance(results, list):
            continue

        for rank, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                continue
            row = dict(base)
            row["rank"] = rank
            for key, value in result.items():
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    input_path = Path("reranked_results.json")
    output_path = Path("reranked_results.csv")

    raw = input_path.read_text(encoding="utf-8")
    records = parse_concatenated_json(raw)
    rows = to_csv_rows(records)
    write_csv(rows, output_path)

    print(f"Parsed blocks: {len(records)}")
    print(f"CSV rows written: {len(rows)}")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
