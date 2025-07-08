import json
from reward_function import reward_function  # 确保你已有的函数可导入

input_file = "arcade-nl2code/arcade_nl2code/annotated_dataset/merged_dataset_new_tasks_cleaned_v2.jsonl"
output_file = "arcade-nl2code/arcade_nl2code/annotated_dataset/merged_dataset_for_ppo_a2c.jsonl"

num_total = 0
num_success = 0

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        num_total += 1
        try:
            item = json.loads(line)
            prompt = item["prompt"]
            completion = item["completion"]

            # 去除 <|endoftext|> 作为 reference
            reference = completion.replace("<|endoftext|>", "").strip()

            # 调用 reward_function 计算分数
            reward = reward_function(prompt, completion, reference)

            # 写入新字段
            item["reference"] = reference
            item["reward"] = float(reward)

            fout.write(json.dumps(item) + "\n")
            num_success += 1

        except Exception as e:
            print(f"❌ Error processing line {num_total}: {e}")
            continue

print(f"\n✅ 数据扩展完成，共处理样本：{num_total}，成功写入：{num_success}")
print(f"📄 输出文件位置：{output_file}")
