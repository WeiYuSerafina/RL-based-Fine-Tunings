import json
import re
import glob
from pathlib import Path
from tqdm import tqdm

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

def split_exercises(instruction_block):
    """
    Splits a prompt block containing multiple exercises into separate ones.
    Returns a list of (instruction_text, question_text) tuples.
    """
    exercises = re.split(r"Exercise\s+\d+", instruction_block)
    outputs = []
    for ex in exercises:
        lines = [
            l.strip().replace("In[ ]:", "")
            for l in ex.strip().splitlines()
            if l.strip() and not l.strip().startswith("In[")
        ]
        if not lines:
            continue

        # ✅ 更强的关键词匹配判断是否是问题行
        problem_lines = [
            l for l in lines
            if re.search(r'(how|what|which|calculate|problem|create|plot|count|find|rank|show)', l.lower())
            or '?' in l
        ]
        if not problem_lines:
            continue

        question = problem_lines[-1]  # last matching question-like line
        instruction_text = f"Convert the following request to code: {question.strip()}"
        outputs.append((instruction_text, "\n".join(lines)))
    return outputs

def extract_context(context):
    """Returns only useful context code lines"""
    if isinstance(context, list):
        lines = context
    else:
        lines = context.split('\n')
    keep = []
    for line in lines:
        if any(x in line for x in ['import ', 'read_csv', '= pd.', '= np.', '= df', 'from '] ):
            keep.append(line.strip())
    return "\n".join(keep)

def is_prompt_context_aligned(prompt, context_block):
    prompt_keywords = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
    context_keywords = set(re.findall(r'\b\w{4,}\b', context_block.lower()))
    overlap = prompt_keywords & context_keywords
    return len(overlap) >= 1  # 可调节容忍度

seen_prompt_bodies = set()

def is_valid_prompt(prompt):
    """判断 prompt 是否为问题句"""
    question_keywords = ['how', 'what', 'which', 'why', 'who', 'calculate', 'problem', '?']
    lower_prompt = prompt.lower()
    return any(keyword in lower_prompt for keyword in question_keywords)

def process_entry(entry):
    raw_prompt = entry.get("prompt", "").strip()
    raw_completion = entry.get("completion", "").strip()

    if not raw_prompt or not raw_completion:
        return []

    # 去除统一开头
    prompt_body = raw_prompt.lower().replace("convert the following request to code:", "").strip()

    # 🚫 跳过重复问题
    if prompt_body in seen_prompt_bodies:
        return []
    seen_prompt_bodies.add(prompt_body)

    # 🚫 跳过 “Let's solve...” 开头的无效 prompt
    if prompt_body.startswith("solution: let's solve"):
        return []

    # 🚫 prompt 不是有效问题（如无 what / how / problem 等关键词）
    if not is_valid_prompt(prompt_body):
        return []

    # Split into multiple instructions (exercise 拆分）
    instruction_blocks = split_exercises(raw_prompt)
    if not instruction_blocks:
        return []

    # 提取 context
    context_block = ""
    if 'Context Code:' in raw_prompt:
        context_block = raw_prompt.split('Context Code:')[-1].strip()
        context_block = extract_context(context_block)
        # 🚫 去除 debug print 或 display
        context_block = '\n'.join([line for line in context_block.splitlines()
                                   if not re.search(r'\b(print|display)\s*\(', line)])

    outputs = []
    for instruction, _ in instruction_blocks:
        prompt = instruction.strip()
        if context_block and not is_prompt_context_aligned(prompt, context_block):
            continue

        if context_block:
            prompt += f"\nContext Code:\n{context_block}"

        completion = raw_completion
        outputs.append({"prompt": prompt, "completion": completion})
    return outputs

def process_file(json_file, output_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        episodes = json.load(f)

    with open(output_file, 'a', encoding='utf-8') as out_f:
        for episode in episodes:
            for turn in episode.get('turns', []):
                prompt = turn.get("input", "").strip()
                code = turn.get("turn", {}).get("code", {}).get("value", "").strip()
                context = turn.get("turn", {}).get("code_context", "")
                if not prompt or not code:
                    continue

                context_str = context if isinstance(context, str) else '\n'.join(context)
                context_str = context_str.replace("In[ ]:", "")

                full_entry = {
                    "prompt": f"Convert the following request to code: {prompt}\nContext Code:\n{context_str}",
                    "completion": code + '\n<|endoftext|>'
                }

                cleaned_samples = process_entry(full_entry)
                for sample in cleaned_samples:
                    out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')

def main():
    input_files = glob.glob('arcade-nl2code/arcade_nl2code/annotated_dataset/dataset/new_tasks/derived_datasets/*.json')
    if len(input_files) == 0:
        print("❌ 没有找到任何 JSON 文件，请检查路径是否正确！")
        return
    else:
        print(f"✅ 找到 {len(input_files)} 个 JSON 文件，开始处理...")

    output_file = 'arcade-nl2code/arcade_nl2code/annotated_dataset/merged_dataset_new_tasks_cleaned.jsonl'
    open(output_file, 'w', encoding='utf-8').close()

    for json_file in tqdm(input_files, desc="Processing files"):
        process_file(json_file, output_file)

    print(f"✅ 处理完成！结果已保存到 {output_file}")
    print(f"📦 总样本数: {sum(1 for _ in open(output_file, 'r', encoding='utf-8'))}")

if __name__ == '__main__':
    main()
