import numpy as np
from memm import MEMM
from hmm import HMM


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
    emis = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]]).T
    trans = np.array([[0.8, 0.2, 0], [0.1, 0.7, 0.2], [0, 0.1, 0.9]])
    r = DiceSeries(3, trans, emis)
    (seq, hidden_state) = r.genSeq(10000)

    m = MEMM(seq)
    m.memm_training()
    print m.memm_viterbi()
    print m.P
    print (m.memm_viterbi() - hidden_state == 0).sum()

    trans = np.array([[0.7, 0.3, 0], [0.1, 0.5, 0.4], [0.3, 0.0, 0.7]])
    emis = np.array([[0.7, 0.7, 0.2], [0.3, 0.3, 0.8]]).T
    h = HMM(3, trans, emis, seq)

    #likelystate = h.hmm_viterbi()

    #print (likelystate == hidden_state).sum()

    #h.hmm_training(1e-10)
    #h.restructure()
    #print h.t
    #print h.e
    #print h.prior
    #print (h.hmm_viterbi() - hidden_state == 0).sum()
