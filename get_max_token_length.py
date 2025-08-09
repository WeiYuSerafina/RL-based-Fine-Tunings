import json
from transformers import GPT2Tokenizer

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

input_path = "arcade-nl2code/arcade_nl2code/annotated_dataset/merged_dataset_new_tasks_cleaned.jsonl"
output_path = "arcade-nl2code/arcade_nl2code/annotated_dataset/filtered_merged_dataset_new_tasks.jsonl"

MAX_TOKENS = 512
MIN_TOKENS = 400
kept_count = 0
skipped_count = 0

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        try:
            item = json.loads(line)
            prompt = item["prompt"].strip()
            completion = item["completion"].strip()
            full_text = f"{prompt}\n{completion}"

            tokens = tokenizer.encode(full_text, add_special_tokens=False)
            token_count = len(tokens)

            if MIN_TOKENS <= token_count <= MAX_TOKENS:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            skipped_count += 1
            print(f"⚠️ Error on line {i}: {e}")

print(f"\n✅ Kept samples: {kept_count}")
print(f"⚠️ Skipped samples (token not in {MIN_TOKENS}-{MAX_TOKENS}): {skipped_count}")
