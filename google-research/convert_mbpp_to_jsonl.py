import json, hashlib
from pathlib import Path

# Optional: List all possible NL field names here
PROMPT_KEYS = ("prompt", "text")

BASE_DIR = Path("/Users/serafinayu/PycharmProjects/nanoGPT-RL/google-research/mbpp")
SRC_FILE = BASE_DIR / "sanitized-mbpp.json"
DST_FILE = BASE_DIR / "sanitized_mbpp_for_nanoGPT.jsonl"

def read_mbpp(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def get_prompt(rec: dict) -> str:
    for k in PROMPT_KEYS:
        if k in rec:
            return rec[k]
    raise KeyError(f"No prompt key ({PROMPT_KEYS}) found in record: {rec}")

def dedup(records):
    seen, deduped = set(), []
    for rec in records:
        key = hashlib.md5((get_prompt(rec) + rec["code"]).encode()).hexdigest()
        if key not in seen:
            deduped.append(rec)
            seen.add(key)
    return deduped

def convert(records, out_path):
    with out_path.open("w", encoding="utf-8") as out_f:
        for r in records:
            prompt     = get_prompt(r).strip() + "\n"        # ➌ 改
            completion = r["code"].strip() + "\n"
            json_line  = {"prompt": prompt, "completion": completion}
            out_f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

def main():
    raw = read_mbpp(SRC_FILE)

    cleaned = [r for r in raw if get_prompt(r).strip() and r["code"].strip()]
    cleaned = dedup(cleaned)

    convert(cleaned, DST_FILE)
    print(f"✓ Conversion completed! {len(cleaned)} samples written → {DST_FILE}")

if __name__ == "__main__":
    main()
