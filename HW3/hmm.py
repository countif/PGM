import numpy as np


class HMM():
    def __init__(self, state_num, trans, emis, seq, prior=np.array([1, 0, 0])):
        self.state_num = state_num
        self.t = trans
        self.e = emis
        self.x = seq
        self.T = len(seq)
        self.prior = prior

        self.alpha = np.zeros(shape=(self.T, self.state_num))
        self.beta = np.zeros(shape=(self.T, self.state_num))
        self.c = np.zeros(shape=(self.T))

        self.hidden_state = np.zeros(shape=(self.T))

    def fowardPropagation(self):
        self.alpha[0] = self.prior * self.e[:, int(self.x[0])].T
        self.c[0] = 1. / self.alpha[0].sum()
        self.alpha[0] *= self.c[0]
        for t in range(1, self.T):
            self.alpha[t] = self.alpha[
                t - 1].dot(self.t) * self.e[:, int(self.x[t])].T
            self.c[t] = 1. / self.alpha[t].sum()
            self.alpha[t] *= self.c[t]

    def backPropagation(self):
        self.beta[self.T - 1] = np.array([1, 1, 1])
        self.beta[self.T - 1] *= self.c[self.T - 1]
        for t in range(self.T - 1, 0, -1):
            self.beta[t - 1] = (self.t *
                                self.e[:, int(self.x[t])].T).dot(self.beta[t])
            self.beta[t - 1] *= self.c[t - 1]

    def return_logP(self):
        return -np.log(self.c).sum()

    def hmm_viterbi(self):
        delta = np.zeros(shape=(self.T, self.state_num))
        phi = np.zeros(shape=(self.T, self.state_num))
        delta[0] = self.prior * self.e[:, int(self.x[0])].T
        delta[0] /= delta[0].sum()
        for t in range(1, self.T):
            for i in range(self.state_num):
                for y in range(self.state_num):
                    if delta[t, i] <= delta[t - 1, y] * self.t[y, i] * self.e[i, int(self.x[t])]:
                        delta[t, i] = delta[t - 1, y] * \
                            self.t[y, i] * self.e[i, int(self.x[t])]
                        phi[t, i] = y
            delta[t] /= delta[t].sum()
        self.hidden_state[
            self.T - 1] = max(range(self.state_num), key=lambda x: delta[self.T - 1, x])
        for t in range(self.T - 1, 0, -1):
            self.hidden_state[t - 1] = phi[t, int(self.hidden_state[t])]
        return self.hidden_state

    def hmm_iteration(self):
        # e-step
        self.backPropagation()
        zeta = np.zeros(shape=(self.T, self.state_num, self.state_num))
        gamma = np.zeros(shape=(self.T, self.state_num))
        for t in range(self.T - 1):
            zeta[t] = self.alpha[t][:, np.newaxis] * self.t * \
                self.e[:, int(self.x[t + 1])].T * self.beta[t + 1]
            zeta[t] = zeta[t] / zeta[t].sum()
            gamma[t] = zeta[t].sum(axis=1)

        # m-step
        self.t = np.asarray(zeta.sum(axis=0) /
                            gamma.sum(axis=0)[:, np.newaxis])
        temp = np.asarray(self.x)
        temp = np.asmatrix(np.vstack((1 - temp, temp)))
        self.e = np.asarray(temp.dot(gamma).T /
                            gamma.sum(axis=0)[:, np.newaxis])
        self.prior = np.asarray(gamma[0])

        self.fowardPropagation()

    def hmm_training(self, epsilon):
        self.fowardPropagation()
        last_logP = -10000
        index = 0
        logP = self.return_logP()
        while abs(logP - last_logP) > epsilon and index < 100:
            last_logP = logP
            index += 1
            self.hmm_iteration()
            logP = self.return_logP()
            print(str(index) + ' iter: logP =' + str(logP))

    def restructure(self):
        a = sorted(range(self.state_num), key=lambda x: self.e[x, 1])
        self.e = self.e[a, :]
        self.t = self.t[a, :]
        self.t = self.t[:, a]
