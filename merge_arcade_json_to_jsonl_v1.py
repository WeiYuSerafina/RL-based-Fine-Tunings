import json
import glob
import re
import ast
from tqdm import tqdm

# Config
input_pattern = 'arcade-nl2code/arcade_nl2code/annotated_dataset/dataset/new_tasks/derived_datasets/*.json'
output_file = 'arcade-nl2code/arcade_nl2code/annotated_dataset/merged_dataset_new_tasks_cleaned_v2.jsonl'
skipped_log_file = 'arcade-nl2code/arcade_nl2code/annotated_dataset/skipped_samples_debug_v2.jsonl'
save_skipped = True

# Syntax check function
def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

# Helpers
def extract_question(text):
    lines = text.splitlines()
    for line in reversed(lines):
        if re.search(r'(how|what|which|why|who|problem|calculate|find|plot|count|create|get|show|return|determine)', line.lower()) or '?' in line:
            return line.strip('# ').strip()
    return None

def clean_context(context):
    lines = context.splitlines() if isinstance(context, str) else context
    keep = []
    for line in lines:
        line = line.strip()

        # Remove comment lines that contain instruction-like text
        if line.startswith("#") and re.search(r'(how|what|which|why|who|problem|calculate|find|plot|count|create|get|show|return|determine|percentage)', line.lower()):
            continue

        # Keep useful code context
        if re.search(r'(import |read_csv|= pd\.|= np\.|from )', line):
            keep.append(line)

    return '\n'.join(keep)

def clean_completion(code):
    lines = code.splitlines()
    keep = []
    for line in lines:
        line = line.strip()
        if line.startswith("import") or "read_csv" in line or line.startswith("#"):
            continue
        keep.append(line)
    return '\n'.join(keep)

def is_valid_instruction(text):
    # Filter out lines that look like code rather than natural-language instructions
    if re.match(r'^(import |df\.|[a-zA-Z0-9_]+ = )', text.strip()):
        return False
    if len(text.split()) < 3:
        return False
    return True

def format_code_block(text):
    """
    Trim trailing spaces, compress excessive blank lines, and preserve indentation.
    """
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    lines = text.splitlines()
    formatted = [line.rstrip() for line in lines]
    return '\n'.join(formatted)

# Main
total_items = 0
skipped_items = 0
kept_items = 0

if save_skipped:
    skipped_out = open(skipped_log_file, 'w', encoding='utf-8')

with open(output_file, 'w', encoding='utf-8') as out_f:
    for file_path in tqdm(glob.glob(input_pattern), desc="Merging JSON files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            episodes = json.load(f)

        for episode in episodes:
            for turn in episode.get("turns", []):
                total_items += 1

                # Prefer the cleaned metadata field if available
                question = turn.get("turn", {}).get("metadata", {}).get("utterance_without_output_spec", "").strip()

                # Fallback: use intent.value and strip the '# Problem:' prefix if present
                if not question:
                    raw_question = turn.get("turn", {}).get("intent", {}).get("value", "").strip()
                    question = re.sub(r"^(?:\s*#\s*Problem:\s*)+", "", raw_question)

                code = turn.get("turn", {}).get("code", {}).get("value", "").strip()
                context = turn.get("turn", {}).get("code_context", "")
                context_cleaned = clean_context(context)
                code_cleaned = clean_completion(code)

                # Validate required fields
                if not question or not code_cleaned:
                    skipped_items += 1
                    if save_skipped:
                        skipped_out.write(json.dumps({
                            "reason": "empty fields",
                            "file": file_path,
                            "question": question,
                            "code": code
                        }) + '\n')
                    continue

                if not is_valid_instruction(question) or len(code_cleaned.split()) < 2:
                    skipped_items += 1
                    if save_skipped:
                        skipped_out.write(json.dumps({
                            "reason": "invalid instruction",
                            "file": file_path,
                            "question": question
                        }) + '\n')
                    continue

                if not is_valid_python(code_cleaned):
                    skipped_items += 1
                    if save_skipped:
                        skipped_out.write(json.dumps({
                            "reason": "invalid python",
                            "file": file_path,
                            "code": code_cleaned
                        }) + '\n')
                    continue

                if not context_cleaned.strip():
                    skipped_items += 1
                    if save_skipped:
                        skipped_out.write(json.dumps({
                            "reason": "empty context",
                            "file": file_path
                        }) + '\n')
                    continue

                context_formatted = format_code_block(context_cleaned)
                code_formatted = format_code_block(code_cleaned)

                item = {
                    "prompt": (
                        "Instruction:\n"
                        f"# Problem: {question.strip()}\n"
                        "Context:\n"
                        f"{context_formatted.strip()}\n"
                        "###\n"
                        "Output:\n"
                    ),
                    "completion": f"{code_formatted.strip()}\n<|endoftext|>"
                }

                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                kept_items += 1

if save_skipped:
    skipped_out.close()
    print(f"Skipped samples are logged to: {skipped_log_file}")

print(f"Merge and cleaning completed. Output file: {output_file}")
print(f"Total samples: {total_items}")
print(f"Skipped samples: {skipped_items}")
print(f"Kept samples: {kept_items}")
print(f"Output file: {output_file}")
