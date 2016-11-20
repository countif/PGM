import numpy as np


class MEMM():
    """docstring for Memm"""

    def __init__(self, seq, state_num=3, obs_num=2):
        self.state_num = state_num
        self.obs_num = obs_num
        self.length = len(seq)
        self.x = seq
        self.y = np.random.randint(0, self.state_num, size=self.length + 1)

        self.lambdas = np.array([1.] * self.state_num * self.state_num *
                                self.obs_num).reshape(self.state_num, self.state_num, self.obs_num)

        self.P = np.random.random(
            (self.state_num, self.state_num, self.obs_num))
        self.P = self.P / self.P.sum(axis=0)

    def cal_P(self):
        for last_s in range(self.state_num):
            for s in range(self.state_num):
                for o in range(self.obs_num):
                    self.P[last_s, s, o] = np.exp(
                        self.lambdas[last_s, s, o])
        z = self.P.sum(axis=1)
        for last_s in range(self.state_num):
            for o in range(self.obs_num):
                self.P[last_s, :, o] = self.P[last_s, :, o] / z[last_s, o]

    def GIS_iteration(self):
        # Note the expectation of features defined here is exactly the
        # Probability
        features = np.array([1.] * self.state_num * self.state_num * self.obs_num).reshape(self.state_num, self.state_num, self.obs_num)
        E = np.array([1.] * self.state_num * self.state_num * self.obs_num).reshape(self.state_num, self.state_num, self.obs_num)
        for i in range(self.length):
            E[int(self.y[i]), :, int(self.x[i])] += self.P[int(self.y[i]), :, int(self.x[i])]
            features[int(self.y[i]), int(self.y[i + 1]), int(self.x[i])] += 1.
        z = features.sum(axis=(1, 2))
        for i in range(self.state_num):
            features[i] = features[i] / z[i]
            E[i] = E[i] / z[i]
        self.lambdas = self.lambdas + np.log(features / E)
        self.cal_P()

    def memm_viterbi(self, prior=np.array([1, 0, 0])):
        delta = np.zeros(shape=(self.length + 1, self.state_num))
        phi = np.zeros(shape=(self.length, self.state_num))
        delta[0] = prior
        for t in range(1, self.length + 1):
            for s in range(self.state_num):
                for last_s in range(self.state_num):
                    if delta[t, s] <= delta[t - 1, last_s] * self.P[last_s, s, int(self.x[t - 1])]:
                        delta[t, s] = delta[t - 1, last_s] * self.P[last_s, s, int(self.x[t - 1])]
                        phi[t - 1, s] = last_s
        self.y[self.length] = max(range(self.state_num), key=lambda x: delta[self.length, x])
        for t in range(self.length, 0, -1):
            self.y[t - 1] = phi[t - 1, int(self.y[t])]
        return self.y[1:]

    def memm_training(self):
        for i in range(100):
            # e-step
            self.memm_viterbi()
            # m-step
            for j in range(5):
                self.GIS_iteration()
