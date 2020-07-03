import numpy as np
import matplotlib.pyplot as plt


ACTION_NUM = 3
EPI_LEN = 1000
EPI_NUM = 2000


class BanditMachine(object):
    def __init__(self, n_arm):
        self.n_arm = n_arm
        self.true_reward = np.random.normal(0, 1, size=self.n_arm)

    def reset(self):
        self.true_reward = np.random.normal(0, 1, size=self.n_arm)

    def get_true_rew(self, action):
        return self.true_reward[action]

    def get_solution(self):
        best_a = np.argmax(self.true_reward)
        return best_a, self.get_true_rew(best_a)

    def action_num(self):
        return self.n_arm

    def step(self, action):
        assert action < self.n_arm
        return np.random.normal(self.true_reward[action], 1)


class EpsilonGreedyAgent(object):
    def __init__(self, eps, action_num):
        assert eps <= 1.0
        self.eps = eps
        self.action_num = action_num
        self.ep_len = 0
        self.q = np.zeros(self.action_num)
        self.chosen_num = np.zeros(self.action_num)

    def reset(self):
        self.ep_len = 0
        self.q = np.zeros(self.action_num)
        self.chosen_num = np.zeros(self.action_num)

    def step(self, a, r):
        assert a < self.action_num
        self.ep_len += 1
        old_q, old_n = self.q[a], self.chosen_num[a]
        new_q = old_q + (r - old_q) / (old_n + 1)
        self.chosen_num[a] += 1
        self.q[a] = new_q

    def get_action(self):
        opt_a = np.argmax(self.q)
        if np.random.uniform(low=0, high=1) > self.eps:
            return opt_a
        else:
            return np.random.randint(0, self.action_num)


def train(env, agent, epi_len, epi_num):
    rewards = np.zeros(epi_len)
    best_agent_a = np.zeros(epi_len)
    max_rewards = 0
    for _ in range(epi_num):
        agent.reset()
        env.reset()
        best_a, max_r = env.get_solution()
        max_rewards += max_r
        for i in range(epi_len):
            a = agent.get_action()
            if a == best_a:
                best_agent_a[i] += 1
            r = env.step(a)
            agent.step(a, r)
            rewards[i] += r
    max_rewards /= epi_num
    for i in range(epi_len):
        rewards[i] /= epi_num
        best_agent_a[i] /= epi_num
    return rewards, best_agent_a, max_rewards


if __name__ == "__main__":
    env = BanditMachine(ACTION_NUM)
    agent_0 = EpsilonGreedyAgent(0, ACTION_NUM)
    agent_0_1 = EpsilonGreedyAgent(0.1, ACTION_NUM)
    agent_0_01 = EpsilonGreedyAgent(0.01, ACTION_NUM)
    rewards_0, best_agent_a_0, max_rewards_0 = train(env, agent_0, EPI_LEN, EPI_NUM)
    rewards_0_1, best_agent_a_0_1, max_rewards_0_1 = train(env, agent_0_1, EPI_LEN, EPI_NUM)
    rewards_0_01, best_agent_a_0_01, max_rewards_0_01 = train(env, agent_0_01, EPI_LEN, EPI_NUM)

    # draw
    epis = list(range(EPI_LEN))
    plt.plot(epis, rewards_0, '-', label="e-0")
    plt.plot(epis, rewards_0_1, '-', label="e-0.1")
    plt.plot(epis, rewards_0_01, '-', label="e-0.01")
    print("max rewards", max_rewards_0, max_rewards_0_1, max_rewards_0_01)

    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(epis, best_agent_a_0, '-', label="e-0")
    plt.plot(epis, best_agent_a_0_1, '-', label="e-0.1")
    plt.plot(epis, best_agent_a_0_01, '-', label="e-0.01")
    print("max rewards", max_rewards_0, max_rewards_0_1, max_rewards_0_01)

    plt.legend()
    plt.grid(True)
    plt.show()
