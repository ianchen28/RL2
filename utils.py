import numpy as np


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


def softmax(x):
    origin_shape = x.shape
    if len(origin_shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        y = x - tmp.reshape((origin_shape[0], 1))
        y = np.exp(y)
        tmp = np.sum(y, axis=1)
        y /= tmp.reshape((origin_shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        y = x - tmp
        y = np.exp(y)
        tmp = np.sum(y)
        y /= tmp
    return y


if __name__ == "__main__":
    x = np.random.normal(2, 1, size=(3, 4))
    y = softmax(x)
    print(x)
    print(y)
