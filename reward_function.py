import difflib
import timeit
import re

def reward_function(generated_code, reference_code, prompt=None):
    # ---------------------------
    # 1. 正确性：代码与参考代码的相似度（可部分匹配）
    similarity = difflib.SequenceMatcher(None, generated_code.strip(), reference_code.strip()).ratio()
    correctness = similarity  # 0.0 ~ 1.0
    # ---------------------------

    # ---------------------------
    # 2. Prompt 复读惩罚
    if prompt:
        prompt_similarity = difflib.SequenceMatcher(None, generated_code.strip(), prompt.strip()).ratio()
        if prompt_similarity > 0.6:
            return 0.0  # 严重复读 → reward 直接归零
        elif prompt_similarity > 0.4:
            penalty = (prompt_similarity - 0.4) * 2  # 轻微惩罚
            correctness = max(0.0, correctness - penalty)
    # ---------------------------

    # ---------------------------
    # 3. 上下文变量一致性检查（奖励复用 prompt 中出现的变量名）
    context_match_reward = 0.0
    if prompt:
        # 抽取 prompt 中的变量名（简单用 regex 匹配如 df, df_scores 等）
        prompt_vars = set(re.findall(r'\b\w+\b', prompt))
        generated_vars = set(re.findall(r'\b\w+\b', generated_code))
        overlap = prompt_vars & generated_vars
        context_match_reward = min(len(overlap) / 3.0, 1.0)  # 最多 +1 分
    # ---------------------------

    # ---------------------------
    # 4. 惩罚提早结束（出现 <|endoftext|> 或太短）
    early_stop_penalty = 0.0
    if "<|endoftext|>" in generated_code:
        early_stop_penalty += 0.3
    if len(generated_code.strip().split()) < 5:
        early_stop_penalty += 0.3
    # ---------------------------

    # ---------------------------
    # 5. 结构关键 token 奖励（结构性代码元素，如 groupby/mean 等）
    structure_tokens = ['groupby', 'mean', 'count', 'apply', 'reset_index', 'drop']
    structure_reward = (
        sum(1 for tok in structure_tokens if tok in generated_code) / len(structure_tokens)
        if len(structure_tokens) > 0 else 0.0
    )

    # ---------------------------

    # ---------------------------
    # 6. 执行效率奖励（⚠️ 建议暂时关闭，避免梯度不稳定）
    # try:
    #     execution_time = timeit.timeit(lambda: exec(generated_code, {}), number=1)
    #     if execution_time < 0 or execution_time > 10:
    #         efficiency_reward = 0.0
    #     else:
    #         efficiency_reward = max(0.0, 1 - execution_time)
    # except Exception:
    #     efficiency_reward = 0.0

    efficiency_reward = 0.0  # ✅ 暂时关闭 exec 奖励，避免不稳定
    # ---------------------------

    # ---------------------------
    # 7. 可读性奖励（token 越少，越简洁）
    token_length = len(generated_code.strip().split())
    readability_reward = max(0.0, 1 - token_length / 100.0)
    # ---------------------------

    # ---------------------------
    # 7.5 奖励 shaping：鼓励非空/非终止输出
    shaping_bonus = 0.0
    if generated_code.strip() == "" or "<|endoftext|>" in generated_code:
        shaping_bonus = -0.5  # 惩罚空/早停
    else:
        shaping_bonus = 0.2  # 奖励非空内容
    # ---------------------------

    # ---------------------------
    # 8. 最终 reward 汇总前：检查每一项是否为合法 float
    components = [correctness,
                  efficiency_reward,
                  readability_reward,
                  context_match_reward,
                  structure_reward,
                  early_stop_penalty,
                  shaping_bonus]
    for c in components:
        if not isinstance(c, float) or c != c or c == float("inf") or c == float("-inf"):
            print("❌ Invalid component in reward calculation. Returning 0.")
            return 0.0
    # ---------------------------

    # ---------------------------
    # 9. 汇总各项得分
    total_reward = (
        0.4 * correctness +
        0.1 * efficiency_reward +
        0.1 * readability_reward +
        0.15 * context_match_reward +
        0.15 * structure_reward -
        early_stop_penalty +
        shaping_bonus  # 鼓励非空
    )

    # ✅ 加 reward 下限防崩溃（例如为零会导致 log_prob 无穷大）
    if total_reward < 1e-5:
        total_reward = 0.01

    # ✅ 防御性检查：确保 reward 是合法 float 值
    if total_reward != total_reward or total_reward == float("inf") or total_reward == float("-inf"):
        return 0.0

    return round(max(0.0, total_reward), 4)
