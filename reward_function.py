import timeit

"""
Rewards for calculating code:
- Code correctness (whether it matches the reference code)
- Code execution efficiency (running time)
- Code readability (number of tokens)
"""

def reward_function(generated_code, reference_code):
    # Correctness check
    correctness = 1.0 if generated_code.strip() == reference_code.strip() else 0.0

    # Calculus execution time
    try:
        execution_time = timeit.timeit(lambda: exec(generated_code), number = 1)
        efficiency_reward = max(0.0, 1 - execution_time)
    except Exception:
        efficiency_reward = -1.0

    # Calculus readability
    readability_reward = max(0.0, 1 - len(generated_code) / 500.0)

    return correctness + efficiency_reward + readability_reward




