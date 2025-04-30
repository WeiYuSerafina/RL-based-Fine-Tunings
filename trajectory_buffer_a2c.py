class TrajectoryBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, done, log_prob, value):
        """
        将每一个时间步的体验数据存储到 buffer 中
        参数:
            state: 当前状态 (tokenized input tensor)
            action: 当前动作 (tensor, e.g. token id)
            reward: scalar 奖励
            done: bool 表示 episode 是否结束
            log_prob: 动作的 log 概率
            value: 当前状态的 value 预测
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get_all(self):
        """
        返回全部收集的数据，用于训练
        返回:
            tuple: (states, actions, rewards, dones, log_probs, values)
        """
        return (
            self.states,
            self.actions,
            self.rewards,
            self.dones,
            self.log_probs,
            self.values,
        )
