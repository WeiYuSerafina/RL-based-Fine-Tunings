#!/usr/bin/env python
# prepare_mbpp_train.py
"""
One-shot script to produce a cleaned MBPP file for Teacher SFT.

Steps
-----
1. read  mbpp.jsonl          (≈ 974  tasks)
2. read  sanitized-mbpp.json (≈ 427 tasks) → collect IDs to drop
3. drop those 427 tasks from full set
4. basic cleaning: non-empty prompt / code
5. deduplicate by MD5(prompt+code)
6. write prompt / completion JSONL

Usage
-----
python3 google-research/prepare_mbpp_train.py \
  --full      google-research/mbpp/mbpp.jsonl \
  --sanitized google-research/mbpp/sanitized-mbpp.json \
  --out       google-research/mbpp/mbpp_train.jsonl
"""
import argparse, hashlib, json, sys
from pathlib import Path
from typing import List, Dict
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

# Helpers
PROMPT_KEYS = ("prompt", "text")

def get_prompt(rec: Dict) -> str:
    for k in PROMPT_KEYS:
        if k in rec:
            return rec[k]
    raise KeyError(f"No prompt/text in record keys={rec.keys()}")

def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def load_json_or_jsonl(path: Path) -> List[Dict]:
    txt = path.read_text(encoding="utf-8").lstrip()
    return json.loads(txt) if txt.startswith("[") else load_jsonl(path)

# Main pipeline
def collect_sanitized_ids(sanitized_path: Path) -> set:
    recs = load_json_or_jsonl(sanitized_path)
    id_key = "task_id" if "task_id" in recs[0] else "id"
    return {r[id_key] for r in recs}

def basic_clean(rec: Dict) -> bool:                 # ★ 更新
    return get_prompt(rec).strip() and rec.get("code", "").strip()

def deduplicate(records: List[Dict]) -> List[Dict]: # ★ 更新
    seen, out = set(), []
    for r in records:
        key = hashlib.md5((get_prompt(r) + r["code"]).encode()).hexdigest()
        if key not in seen:
            seen.add(key); out.append(r)
    return out

def convert(records: List[Dict], out_path: Path):   # ★ 更新
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "prompt":     get_prompt(r).rstrip() + "\n",
                "completion": r["code"].rstrip()     + "\n"
            }, ensure_ascii=False) + "\n")

def main(full_path: Path, sanitized_path: Path, out_path: Path):
    print(f"→ loading full file:      {full_path}")
    full_records = load_jsonl(full_path)
    print(f"  full records:          {len(full_records):,}")

    print(f"→ loading sanitized set: {sanitized_path}")
    bad_ids = collect_sanitized_ids(sanitized_path)
    print(f"  sanitized ids to drop: {len(bad_ids):,}")

    filtered = [r for r in full_records if r["task_id"] not in bad_ids]
    print(f"  after drop:            {len(filtered):,}")

    cleaned = [r for r in filtered if basic_clean(r)]
    print(f"  after basic clean:     {len(cleaned):,}")

    deduped = deduplicate(cleaned)
    print(f"  after dedup:           {len(deduped):,}")

    convert(deduped, out_path)
    print(f"\n✓ DONE! wrote {len(deduped):,} samples → {out_path}")

# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full",      required=True, help="path to mbpp.jsonl (≈974)")
    ap.add_argument("--sanitized", required=True, help="path to sanitized-mbpp.json (≈427)")
    ap.add_argument("--out",       default="mbpp_train.jsonl",
                    help="output cleaned jsonl (prompt/completion)")
    args = ap.parse_args()
    main(Path(args.full), Path(args.sanitized), Path(args.out))
