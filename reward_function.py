import difflib
import numpy as np
import re

def reward_function(generated_code, reference_code, prompt=None):
    # 1. Correctness: The similarity of the code to the reference code
    similarity = difflib.SequenceMatcher(None, generated_code.strip(), reference_code.strip()).ratio()
    correctness = similarity  # 0.0 ~ 1.0

    # 2. Prompt repeat penalty
    if prompt:
        prompt_similarity = difflib.SequenceMatcher(None, generated_code.strip(), prompt.strip()).ratio()
        if prompt_similarity > 0.6:
            return 0.0  # Severe repetition → reward is reset to zero
        elif prompt_similarity > 0.4:
            penalty = (prompt_similarity - 0.4) * 2  # Minor penalty
            correctness = max(0.0, correctness - penalty)

    # 3. Context variable consistency check (rewards reuse of variable names that appear in prompt)
    context_match_reward = 0.0
    if prompt:
        # Extract the variable name in the prompt (simply use regex to match such as df, df_scores, etc.)
        prompt_vars = set(re.findall(r'\b\w+\b', prompt))
        generated_vars = set(re.findall(r'\b\w+\b', generated_code))
        overlap = prompt_vars & generated_vars
        context_match_reward = min(len(overlap) / 3.0, 1.0)  # Maximum +1

    # 4. Penalizes premature ending (<|endoftext|> appears or is too short)
    early_stop_penalty = 0.0
    if "<|endoftext|>" in generated_code:
        early_stop_penalty += 0.3
    if len(generated_code.strip().split()) < 5:
        early_stop_penalty += 0.3

    # 5. Structural key token rewards (structural code elements such as groupby/mean, etc.)
    structure_tokens = ['groupby', 'mean', 'count', 'apply', 'reset_index', 'drop']
    structure_reward = (
        sum(1 for tok in structure_tokens if tok in generated_code) / len(structure_tokens)
        if len(structure_tokens) > 0 else 0.0
    )

    # 6. Execution efficiency reward (It is recommended to temporarily disable it to avoid gradient instability)
    # try:
    #     execution_time = timeit.timeit(lambda: exec(generated_code, {}), number=1)
    #     if execution_time < 0 or execution_time > 10:
    #         efficiency_reward = 0.0
    #     else:
    #         efficiency_reward = max(0.0, 1 - execution_time)
    # except Exception:
    #     efficiency_reward = 0.0
    efficiency_reward = 0.0  # Temporarily disable exec rewards to avoid instability

    # 7. Readability bonus (fewer tokens, more concise)
    token_length = len(generated_code.strip().split())
    readability_reward = max(0.0, 1 - token_length / 100.0)

    # 7.5 Shaping bonus
    shaping_bonus = 0.0
    if generated_code.strip() == "" or "<|endoftext|>" in generated_code:
        shaping_bonus = -0.5  # Penalize empty/early stop
    else:
        shaping_bonus = 0.2  # Reward non-empty content

    # 8. Before the final reward is aggregated: check whether each item is a valid float
    components = [correctness,
                  efficiency_reward,
                  readability_reward,
                  context_match_reward,
                  structure_reward,
                  early_stop_penalty,
                  shaping_bonus]
    for c in components:
        if not isinstance(c, float) or c != c or c == float("inf") or c == float("-inf"):
            print("Invalid component in reward calculation. Returning 0.")
            return 0.0

    # 9. Summarize the scores
    total_reward = (
        0.4 * correctness +
        0.1 * efficiency_reward +
        0.1 * readability_reward +
        0.15 * context_match_reward +
        0.15 * structure_reward -
        early_stop_penalty +
        shaping_bonus  # Encourage non-empty
    )

    # Running mean/std normalization
    EPS = 1e-8
    global running_mean, running_M2, running_count
    if 'running_mean' not in globals():
        running_mean, running_M2, running_count = 0.0, 0.0, 0  # First initialization

    running_count += 1
    delta = total_reward - running_mean
    running_mean += delta / running_count
    running_M2 += delta * (total_reward - running_mean)
    reward_std = ((running_M2 / max(running_count, 1)) ** 0.5) + EPS
    total_reward = (total_reward - running_mean) / reward_std

    # tanh is compressed to (0,1) + floor = 0.01
    total_reward = (np.tanh(total_reward) + 1) / 2  # (-1,1)→(0,1)
    total_reward = max(total_reward, 0.01)  # Fixed lower limit 0.01

    # Clip now only needs to prevent over-limit
    total_reward = float(np.clip(total_reward, 0.01, 1.0))
    if np.isnan(total_reward) or np.isinf(total_reward):
        return 0.0

    return round(total_reward, 4)
