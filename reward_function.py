import difflib
import timeit

def reward_function(generated_code, reference_code, prompt=None):
    # --- 1. Correctness: 基于代码相似度（支持部分正确）
    similarity = difflib.SequenceMatcher(None, generated_code.strip(), reference_code.strip()).ratio()
    correctness = similarity  # between 0.0 and 1.0

    # --- 2. Prompt Copy Penalty (强惩罚复读 prompt)
    if prompt:
        prompt_similarity = difflib.SequenceMatcher(None, generated_code.strip(), prompt.strip()).ratio()
        print(f"prompt_similarity: {prompt_similarity:.4f}")  # debug

        # 如果生成内容与 prompt 的相似度过高，直接重罚
        if prompt_similarity > 0.6:
            print(f"Prompt copy detected. Similarity: {prompt_similarity:.2f}")
            return 0.0  # 完全不给 reward
        elif prompt_similarity > 0.4:
            penalty = (prompt_similarity - 0.4) * 2  # 线性惩罚（最大 0.4）
            correctness = max(0.0, correctness - penalty)

    # --- 3. Execution efficiency
    try:
        execution_time = timeit.timeit(lambda: exec(generated_code, {}), number=1)
        efficiency_reward = max(0.0, 1 - execution_time)
    except Exception:
        efficiency_reward = 0.0

    # --- 4. Readability
    token_length = len(generated_code.strip().split())
    readability_reward = max(0.0, 1 - token_length / 100.0)

    # --- 5. Total reward
    total_reward = 0.6 * correctness + 0.2 * efficiency_reward + 0.2 * readability_reward
    return round(total_reward, 4)
