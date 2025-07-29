import difflib
import numpy as np
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

    # === ⭐ running mean/std 归一化 =========================
    EPS = 1e-8
    global running_mean, running_M2, running_count
    if 'running_mean' not in globals():
        running_mean, running_M2, running_count = 0.0, 0.0, 0  # 首次初始化

    running_count += 1
    delta = total_reward - running_mean
    running_mean += delta / running_count
    running_M2 += delta * (total_reward - running_mean)
    reward_std = ((running_M2 / max(running_count, 1)) ** 0.5) + EPS
    total_reward = (total_reward - running_mean) / reward_std
    # =====================================================

    # === 🔧 NEW: tanh 压缩到 (0,1) + floor = 0.01 =========
    total_reward = (np.tanh(total_reward) + 1) / 2  # (-1,1)→(0,1)
    total_reward = max(total_reward, 0.01)  # 固定下限 0.01
    # =====================================================
    # ⭐⭐ 放大奖励信号（可调 2~5）
    # total_reward *= 5.0  # <—— 新增这一行
    # -------------------------------------------------------
    # 🔧 UPDATE: clip 现在只需防超上限；不会再出现负值
    total_reward = float(np.clip(total_reward, 0.01, 1.0))
    if np.isnan(total_reward) or np.isinf(total_reward):
        return 0.0

    return round(total_reward, 4)
