import json
import glob
import re
import ast
from tqdm import tqdm

# === CONFIG ===
input_pattern = 'arcade-nl2code/arcade_nl2code/annotated_dataset/dataset/new_tasks/derived_datasets/*.json'
output_file = 'arcade-nl2code/arcade_nl2code/annotated_dataset/merged_dataset_new_tasks_cleaned_v1.jsonl'
skipped_log_file = 'arcade-nl2code/arcade_nl2code/annotated_dataset/skipped_samples_debug.jsonl'
save_skipped = True

# === SYNTAX CHECK FUNCTION ===
def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

# === HELPERS ===
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

        # 去除包含问题指令的注释行
        if line.startswith("#") and re.search(r'(how|what|which|why|who|problem|calculate|find|plot|count|create|get|show|return|determine|percentage)', line.lower()):
            continue

        # 保留有用的代码上下文
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
    if re.match(r'^(import |df\.|[a-zA-Z0-9_]+ = )', text.strip()):
        return False
    if len(text.split()) < 3:
        return False
    return True

def format_code_block(text):
    """
    清理尾部空格、压缩多余空行，保留原始缩进结构。
    """
    text = re.sub(r'\n{3,}', '\n\n', text.strip())  # 多余空行压缩为双换行
    lines = text.splitlines()
    formatted = [line.rstrip() for line in lines]  # 保留前导空格，只清理行尾空格
    return '\n'.join(formatted)

# === MAIN ===
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
                question = turn.get("turn", {}).get("intent", {}).get("value", "").strip()
                code = turn.get("turn", {}).get("code", {}).get("value", "").strip()
                context = turn.get("turn", {}).get("code_context", "")
                context_cleaned = clean_context(context)
                code_cleaned = clean_completion(code)

                # 清洗逻辑
                if not question or not code_cleaned:
                    skipped_items += 1
                    if save_skipped:
                        skipped_out.write(json.dumps({"reason": "empty fields", "file": file_path, "question": question, "code": code}) + '\n')
                    continue

                if not is_valid_instruction(question) or len(code_cleaned.split()) < 2:
                    skipped_items += 1
                    if save_skipped:
                        skipped_out.write(json.dumps({"reason": "invalid instruction", "file": file_path, "question": question}) + '\n')
                    continue

                if not is_valid_python(code_cleaned):
                    skipped_items += 1
                    if save_skipped:
                        skipped_out.write(json.dumps({"reason": "invalid python", "file": file_path, "code": code_cleaned}) + '\n')
                    continue

                if not context_cleaned.strip():
                    skipped_items += 1
                    if save_skipped:
                        skipped_out.write(json.dumps({"reason": "empty context", "file": file_path}) + '\n')
                    continue

                context_formatted = format_code_block(context_cleaned)
                code_formatted = format_code_block(code_cleaned)

                item = {
                    "prompt": f"Instruction: \n{question}\nContext:\n{context_formatted}",
                    "completion": f"{code_formatted}\n<|endoftext|>"
                }
                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                kept_items += 1

if save_skipped:
    skipped_out.close()
    print(f"🪵 跳过样本已记录在: {skipped_log_file}")

print(f"✅ 合并并清洗完成，输出文件: {output_file}")
print(f"\n✨ 总样本数: {total_items}")
print(f"🚫 跳过样本数: {skipped_items}")
print(f"✅ 保留有效样本: {kept_items}")
print(f"📄 输出文件: {output_file}")
