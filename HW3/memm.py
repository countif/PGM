import numpy as np


class MEMM():
    def __init__(self, seq, p=np.random.random(size=(3, 3, 2)), state_num=3, obs_num=2):
        self.state_num = state_num
        self.obs_num = obs_num
        self.x = np.append([0.], seq)
        self.T = len(self.x)
        self.y = np.random.randint(0, 3, size=self.T)
        # P(y_(t+1)|y_t, x)
        self.P = p
        for s in range(self.state_num):
            self.P[s] = self.P[s] / self.P[s].sum(axis=(0))
        # record trans features
        self.features = np.zeros(
            shape=(self.state_num, self.state_num, self.obs_num))
        self.lambdas = np.zeros(
            shape=(self.state_num, self.state_num, self.obs_num))

    def cal_features(self):
        self.features = np.array([1.] * self.state_num * self.state_num *
                                 self.obs_num).reshape(self.state_num, self.state_num, self.obs_num)
        for i in range(1, self.T):
            self.features[int(self.y[i - 1]), int(self.y[i]),
                          int(self.x[i])] += 1.

    def cal_P(self):
        self.P = np.exp(self.lambdas)
        for s in range(self.state_num):
            self.P[s] = self.P[s] / self.P[s].sum(axis=(0))

    def GIS_training(self, mixed_time=5):
        # initiate lambdas
        self.lambdas = np.array([1.] * self.state_num * self.state_num *
                                self.obs_num).reshape(self.state_num, self.state_num, self.obs_num)
        # calculate F_a
        self.cal_features()
        F = np.zeros(shape=(self.state_num, self.state_num, self.obs_num))
        for s in range(self.state_num):
            F[s] = self.features[s] / self.features[s].sum()
        # iteration
        E = np.zeros(shape=(self.state_num, self.state_num, self.obs_num))
        for i in range(mixed_time):
            self.cal_P()
            for s in range(self.state_num):
                E[s] = self.P[s] * self.features[s].sum(
                    axis=1)[:, np.newaxis] / self.features[s].sum()
            self.lambdas = self.lambdas - np.log(F / E)
        self.cal_P()

    # generate a new series of y's by gibbs sampling
    def gibbs_sampling(self):
        for t in range(1, self.T - 1):
            p = self.P[int(self.y[t - 1]), :, int(self.x[t])] * \
                self.P[:, int(self.y[t + 1]), int(self.x[t + 1])]
            self.y[t] = np.random.choice(self.state_num, 1, p=p / sum(p))

        self.y[self.T - 1] = np.random.choice(self.state_num, 1, p=self.P[
                                              int(self.y[self.T - 2]), :, int(self.x[self.T - 1])])

    def gibbs_estimation(self):
        self.P = np.zeros(shape=(self.state_num, self.state_num, self.obs_num))
        self.P += 0.01
        for t in range(1, self.T):
            self.P[int(self.y[t - 1]), int(self.y[t]), int(self.x[t])] += 1
        for s in range(self.state_num):
            self.P[s] = self.P[s] / self.P[s].sum(axis=(0))

    def restructure(self):
        e = np.zeros(self.state_num)
        for t in range(1, self.T):
            e[int(self.y[t])] += self.x[t]
        a = sorted(range(self.state_num), key=lambda x: e[x])
        self.P = self.P[a, :]
        for s in range(self.state_num):
            self.P[s] = self.P[s, a, :]

    def memm_viterbi(self, prior=np.array([1, 0, 0])):
        delta = np.zeros(shape=(self.T, self.state_num))
        phi = np.zeros(shape=(self.T, self.state_num))
        delta[0] = prior
        for t in range(1, self.T):
            for s in range(self.state_num):
                for i in range(self.state_num):
                    if delta[t, s] < delta[t - 1, i] * self.P[i, s, int(self.x[t])]:
                        delta[t, s] = delta[t - 1, i] * \
                            self.P[i, s, int(self.x[t])]
                        phi[t - 1, s] = i
        self.y[self.T - 1] = max(range(self.state_num),
                                 key=lambda x: delta[self.T - 1, x])
        for t in range(self.T - 1, 0, -1):
            self.y[t - 1] = phi[t, int(self.y[t])]
        return self.y[1:]

    def memm_training(self, mixed_time=5, max_iteration=100):
        for i in range(max_iteration):
            self.GIS_training()
            for x in range(mixed_time):
                self.gibbs_sampling()


class DiceSeries():
    def __init__(self, state_num, trans, emis):
        self.state_num = state_num
        self.trans = trans
        self.emis = emis
        x, self.ob_num = emis.shape
        assert((x, x) == trans.shape)

    def genSeq(self, length):
        hidden_state = np.zeros(shape=length)
        seq = np.zeros(shape=length)
        prob = np.zeros(shape=self.state_num)
        prob[0] = 1
        for i in range(length):
            choice = np.random.choice(self.state_num, 1, p=prob)
            hidden_state[i] = choice
            seq[i] = np.random.choice(self.ob_num, 1, p=emis[choice, :][0])
            prob = self.trans[choice, :][0]
        return (seq, hidden_state)


if __name__ == "__main__":
    trans = np.array([[0.8, 0.2, 0], [0.1, 0.7, 0.2], [0.1, 0, 0.9]])
    emis = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]]).T
    r = DiceSeries(3, trans, emis)
    (seq, hidden_state) = r.genSeq(10000)

    p = np.zeros(shape=(3, 3, 2))
    for i in range(3):
        for j in range(3):
            for t in range(2):
                p[i, j, t] = trans[i, j] * emis[j, t]
    m = MEMM(seq)
    likelystate = m.memm_viterbi()
    print (likelystate == hidden_state).sum(
    )
    m.memm_training()
    m.restructure()
    print (m.memm_viterbi() - hidden_state == 0).sum()
