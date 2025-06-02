import os
import json
from glob import glob

def convert_all_new_tasks(new_tasks_dir, output_path):
    all_json_files = glob(os.path.join(new_tasks_dir, "*.json"))
    print(f"📦 共找到 {len(all_json_files)} 个 JSON 文件，开始处理...")

    count_total, count_written = 0, 0
    bad_sample_count = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for file_path in all_json_files:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)

                    if isinstance(data, list):
                        for entry in data:
                            for turn in entry.get("turns", []):
                                turn_data = turn.get("turn", {})
                                instruction = turn_data.get("intent", {}).get("value", "").strip()
                                code = turn_data.get("code", {}).get("value", "").strip()
                                context = turn_data.get("context", {}).get("value", "").strip()

                                # ✅ 严格校验样本结构
                                if not instruction or not code:
                                    bad_sample_count += 1
                                    continue

                                if code == "<|endoftext|>":
                                    bad_sample_count += 1
                                    continue

                                if len(code.splitlines()) < 2:
                                    bad_sample_count += 1
                                    continue

                                if not context:
                                    context = "# No additional context"

                                record = {
                                    "instruction": instruction,
                                    "context": context,
                                    "solution": code + "<|endoftext|>"
                                }
                                fout.write(json.dumps(record) + "\n")
                                count_written += 1

                        count_total += len(data)

                except Exception as e:
                    print(f"❌ 处理失败: {file_path}, 错误: {e}")

    print(f"\n✅ 总共处理 {count_total} 条任务，写入 {count_written} 条有效样本到 {output_path}")
    print(f"⚠️  跳过了 {bad_sample_count} 条无效样本（代码为空、只有 endoftext 或太短）")

# 使用方式
if __name__ == "__main__":
    new_tasks_dir = "arcade-nl2code/arcade_nl2code/annotated_dataset/dataset/new_tasks/derived_datasets"
    output_path = "arcade-nl2code/arcade_nl2code/annotated_dataset/converted_new_tasks.jsonl"
    convert_all_new_tasks(new_tasks_dir, output_path)
