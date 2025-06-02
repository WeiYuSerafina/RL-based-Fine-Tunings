import json

# 输入输出路径
input_file = "arcade-nl2code/arcade_nl2code/annotated_dataset/converted_new_tasks.jsonl"
output_file = "arcade-nl2code/arcade_nl2code/annotated_dataset/converted_new_tasks_top300.jsonl"

# ✅ 自定义关键词（可根据需要扩展）
keywords = [
    "pandas", "numpy", "astype", "float", "int", "extract", "replace", "split",
    "groupby", "merge", "dropna", "fillna", "sort_values", "loc", "iloc",
    "if", "else", "def", "return", "for", "in", "apply", "lambda"
]

def contains_keywords(text, keywords):
    return any(keyword in text for keyword in keywords)

selected_samples = []
with open(input_file, "r", encoding="utf-8") as fin:
    for line in fin:
        try:
            record = json.loads(line)
            completion = record.get("completion", "")
            if contains_keywords(completion, keywords):
                selected_samples.append(record)
                if len(selected_samples) >= 300:
                    break
        except json.JSONDecodeError:
            continue

# 写入结果
with open(output_file, "w", encoding="utf-8") as fout:
    for item in selected_samples:
        fout.write(json.dumps(item) + "\n")

print(f"✅ 筛选完成，共保存 {len(selected_samples)} 条样本到 {output_file}")
