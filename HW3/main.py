import numpy as np
# from memm import MEMM
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


def test(trans, emis, seq):
    state_est = np.zeros(shape=3)
    trans_est = np.zeros(shape=(2, 3, 3))
    emis_est = np.zeros(shape=(2, 3))
    elapsed_time = np.zeros(shape=2)

    h = HMM(3, trans, emis, seq)
    state_est[2] = (h.hmm_viterbi() == hidden_state).sum()

    # parameter:mixted_time, epsilon, max_iteration
    elapsed_time[0] = h.hmm_training_gibbs()
    h.restructure()
    state_est[0] = (h.hmm_viterbi() == hidden_state).sum()
    trans_est[0] = h.t
    emis_est[0] = h.e[:, 1]

    h = HMM(3, trans, emis, seq)
    # parameter:mixted_time, epsilon, max_iteration
    elapsed_time[1] = h.hmm_training()
    h.restructure()
    state_est[1] = (h.hmm_viterbi() == hidden_state).sum()
    trans_est[1] = h.t
    emis_est[1] = h.e[:, 1]

    return state_est, trans_est, emis_est, elapsed_time

if __name__ == "__main__":
    trans = np.array([[0.8, 0.2, 0], [0.1, 0.7, 0.2], [0.1, 0, 0.9]])
    emis = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]]).T
    r = DiceSeries(3, trans, emis)
    (seq, hidden_state) = r.genSeq(1000)

    state_est = np.zeros(shape=(1000, 3))
    trans_est = np.zeros(shape=(1000, 2, 3, 3))
    emis_est = np.zeros(shape=(1000, 2, 3))
    elapsed_time = np.zeros(shape=(1000, 2))

    outfile_name = '1000-dices-1'

    for i in range(1000):
        if (i + 1) % 1 == 0:
            np.savez(outfile_name, state_est=state_est, trans_est=trans_est,
                     emis_est=emis_est, elapsed_time=elapsed_time)

        trans = np.random.random(size=(3, 3))
        emis = np.random.random(size=(3, 2))
        trans = trans / trans.sum(axis=1)[:, np.newaxis]
        emis = emis / emis.sum(axis=1)[:, np.newaxis]

        state_est[i], trans_est[i], emis_est[
            i], elapsed_time[i] = test(trans, emis, seq)
