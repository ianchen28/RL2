import numpy as np
import utils
import multiprocessing
import queue

import matplotlib.pyplot as plt


ACTION_NUM = 10
EPI_LEN = 1000
EPI_NUM = 2000
NUM_CORE = multiprocessing.cpu_count()


class GradientAgent(object):
    def __init__(self, action_num, alpha, with_baseline=True):
        assert 0 <= alpha <= 1
        self.alpha = alpha
        self.action_num = action_num
        self.ep_len = 0
        self.h_a = np.zeros(action_num)
        self.r_bar = 0
        self.pi_a = utils.softmax(self.h_a)
        self.with_baseline = with_baseline

    def reset(self):
        self.ep_len = 0
        self.h_a = np.zeros(self.action_num)
        self.r_bar = 0
        self.pi_a = utils.softmax(self.h_a)

    def step(self, a, r):
        assert a < self.action_num
        self.ep_len += 1
        if self.with_baseline:
            self.r_bar = self.r_bar + (r - self.r_bar) / self.ep_len
        for action in range(self.action_num):
            if action == a:
                self.h_a[action] = self.h_a[action] + self.alpha * (r - self.r_bar) * (1 - self.pi_a[action])
            else:
                self.h_a[action] = self.h_a[action] - self.alpha * (r - self.r_bar) * self.pi_a[action]
        self.pi_a = utils.softmax(self.h_a)

    def get_action(self):
        return np.random.choice(self.action_num, 1, p=self.pi_a)


def task(alpha, with_baseline, epi_len):
    agent = GradientAgent(ACTION_NUM, alpha=alpha, with_baseline=with_baseline)
    env = utils.BanditMachine(ACTION_NUM, r_mean=4)
    best_a, max_r = env.get_solution()
    best_agent_a = np.zeros(epi_len)
    rewards = np.zeros(epi_len)
    for i in range(epi_len):
        a = agent.get_action()
        if a == best_a:
            best_agent_a[i] += 1
        r = env.step(a)
        agent.step(a, r)
        rewards[i] += r
    return rewards, best_agent_a, max_r


def train(alpha, with_baseline, epi_len, epi_num):
    rewards_sum = np.zeros(EPI_LEN)
    best_agent_a_sum = np.zeros(EPI_LEN)
    max_rewards_sum = 0

    pool = multiprocessing.Pool(NUM_CORE - 1)
    results = []
    for i in range(epi_num):
        result = pool.apply_async(
            func=task,
            args=(alpha, with_baseline, epi_len))
        results.append(result)
    pool.close()
    pool.join()
    for result in results:
        rewards, best_agent_a, max_r = result.get()
        for i in range(EPI_LEN):
            rewards_sum[i] += rewards[i]
            best_agent_a_sum[i] += best_agent_a[i]
        max_rewards_sum += max_r

    max_rewards_sum /= epi_num
    for i in range(epi_len):
        rewards_sum[i] /= epi_num
        best_agent_a_sum[i] /= epi_num
    return rewards_sum, best_agent_a_sum, max_rewards_sum


if __name__ == "__main__":
    rewards_0_1, best_agent_a_0_1, max_rewards_0_1 = \
        train(alpha=0.1, with_baseline=True, epi_len=EPI_LEN, epi_num=EPI_NUM)
    rewards_0_4, best_agent_a_0_4, max_rewards_0_4 = \
        train(alpha=0.4, with_baseline=True, epi_len=EPI_LEN, epi_num=EPI_NUM)
    rewards_0_1_no_r_bar, best_agent_a_0_1_no_r_bar, max_rewards_0_1_no_r_bar = \
        train(alpha=0.1, with_baseline=False, epi_len=EPI_LEN, epi_num=EPI_NUM)
    rewards_0_4_no_r_bar, best_agent_a_0_4_no_r_bar, max_rewards_0_4_no_r_bar = \
        train(alpha=0.4, with_baseline=False, epi_len=EPI_LEN, epi_num=EPI_NUM)

    # draw
    epis = list(range(EPI_LEN))
    plt.plot(epis, rewards_0_1, '-', label="a-0.1")
    plt.plot(epis, rewards_0_4, '-', label="a-0.4")
    plt.plot(epis, rewards_0_1_no_r_bar, '-', label="a-0.1-no_bar")
    plt.plot(epis, rewards_0_4_no_r_bar, '-', label="a-0.4-no_bar")

    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(epis, best_agent_a_0_1, '-', label="a-0.1")
    plt.plot(epis, best_agent_a_0_4, '-', label="a-0.4")
    plt.plot(epis, best_agent_a_0_1_no_r_bar, '-', label="a-0.1-no_bar")
    plt.plot(epis, best_agent_a_0_4_no_r_bar, '-', label="a-0.4-no_bar")

    plt.legend()
    plt.grid(True)
    plt.show()
