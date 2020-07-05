import numpy as np
import utils

import matplotlib.pyplot as plt


ACTION_NUM = 10
EPI_LEN = 1000
EPI_NUM = 200


class GradientAgent(object):
    def __init__(self, action_num, alpha=None):
        self.alpha = alpha
        self.action_num = action_num
        self.ep_len = 0
        self.h_a = np.zeros(action_num)
        self.r_bar = 0
        self.pi_a = utils.softmax(self.h_a)

    def reset(self):
        self.ep_len = 0
        self.h_a = np.zeros(self.action_num)
        self.r_bar = 0
        self.pi_a = utils.softmax(self.h_a)

    def step(self, a, r):
        assert a < self.action_num
        self.ep_len += 1
        self.r_bar = self.r_bar + (r - self.r_bar) / self.ep_len
        for action in range(self.action_num):
            if action == a:
                self.h_a[action] = self.h_a[action] + self.alpha * (r - self.r_bar) * (1 - self.pi_a[action])
            else:
                self.h_a[action] = self.h_a[action] + self.alpha * (r - self.r_bar) * self.pi_a[action]
        self.pi_a = utils.softmax(self.h_a)

    def get_action(self):
        return np.random.choice(self.action_num, 1, p=self.pi_a)


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
    env = utils.BanditMachine(ACTION_NUM)
    agent_0_1 = GradientAgent(ACTION_NUM, alpha=0.1)
    agent_0_4 = GradientAgent(ACTION_NUM, alpha=0.4)
    rewards_0_1, best_agent_a_0_1, max_rewards_0_1 = train(env, agent_0_1, EPI_LEN, EPI_NUM)
    rewards_0_4, best_agent_a_0_4, max_rewards_0_4 = train(env, agent_0_4, EPI_LEN, EPI_NUM)

    # draw
    epis = list(range(EPI_LEN))
    plt.plot(epis, rewards_0_1, '-', label="e-0.1")
    plt.plot(epis, rewards_0_4, '-', label="e-0.4")
    print("max rewards", max_rewards_0_1, max_rewards_0_4)

    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(epis, best_agent_a_0_1, '-', label="e-0.1")
    plt.plot(epis, best_agent_a_0_4, '-', label="e-0.4")
    print("max rewards", max_rewards_0_1, max_rewards_0_4)

    plt.legend()
    plt.grid(True)
    plt.show()
