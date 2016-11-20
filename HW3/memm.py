import numpy as np


class MEMM():
    """docstring for Memm"""

    def __init__(self, seq, state_num=3, obs_num=2):
        self.state_num = state_num
        self.obs_num = obs_num
        self.length = len(seq)
        self.x = seq
        self.y = np.random.randint(0, 3, size=self.length + 1)
        # record trans features
        self.features_trans = np.zeros(shape=(self.state_num, self.state_num))
        self.lambda_trans = np.array(
            [1.] * self.state_num * self.state_num).reshape(self.state_num, self.state_num)
        self.features_emis = np.zeros(shape=(self.state_num, self.obs_num))
        self.lambda_emis = np.array(
            [1.] * self.state_num * self.obs_num).reshape(self.state_num, self.obs_num)
        # P(y_(t+1)|y_t, x)
        self.P = np.random.random(
            (self.state_num, self.state_num, self.obs_num))
        self.P = self.P / self.P.sum(axis=0)

    def cal_features(self):
        for i in range(self.length):
            self.features_trans[int(self.y[i]), int(self.y[i + 1])] += 1.
            self.features_emis[int(self.y[i + 1]), int(self.x[i])] += 1.
        self.features_trans = self.features_trans / self.length
        self.features_emis = self.features_emis / self.length

    def cal_P(self):
        for s in range(self.state_num):
            for i in range(self.state_num):
                for o in range(self.obs_num):
                    self.P[s, i, o] = np.exp(
                        self.lambda_trans[i, s] + self.lambda_emis[s, o])
        self.P = self.P / self.P.sum(axis=0)
        # print self.P

    def GIS_iteration(self):
        expectation_trans = np.zeros(shape=(self.state_num, self.state_num))
        expectation_emis = np.zeros(shape=(self.state_num, self.obs_num))
        for i in range(self.length):
            expectation_trans[int(self.y[i]), :] += \
                self.P[:, int(self.y[i]), int(self.x[i])].T
            expectation_emis[:, int(self.x[i])] += \
                self.P[:, int(self.y[i]), int(self.x[i])]
        #expectation_trans = expectation_trans / self.length
        #expectation_emis = expectation_emis / self.length
        print expectation_trans
        self.lambda_trans = self.lambda_trans + 0.5 * \
            np.log(self.features_trans / expectation_trans)
        self.lambda_emis = self.lambda_emis + 0.5 * \
            np.log(self.features_emis / expectation_emis)
        self.cal_P()

    def memm_viterbi(self, prior=np.array([1, 0, 0])):
        delta = np.zeros(shape=(self.length + 1, self.state_num))
        phi = np.zeros(shape=(self.length, self.state_num))
        delta[0] = prior
        for t in range(1, self.length + 1):
            for s in range(self.state_num):
                for i in range(self.state_num):
                    if delta[t, s] <= delta[t - 1, i] * self.P[s, i, int(self.x[t - 1])]:
                        delta[t, s] = delta[t - 1, i] * \
                            self.P[s, i, int(self.x[t - 1])]
                        phi[t - 1, s] = i
        self.y[self.length] = max(
            range(self.state_num), key=lambda x: delta[self.length, x])
        for t in range(self.length, 0, -1):
            self.y[t - 1] = phi[t - 1, int(self.y[t])]
        return self.y

    def memm_training(self):
        for i in range(2):
            # e-step
            self.memm_viterbi()
            self.cal_features()
            # m-step
            self.GIS_iteration()
        # print self.memm_viterbi()
