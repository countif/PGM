import numpy as np
outfile_name = '100-dices-1.npz'
npzfile = np.load(outfile_name)
print npzfile['state_est'][0, :]
print npzfile['trans_est'][0, 0, :]
print npzfile['trans_est'][0, 1, :]
print npzfile['emis_est'][0, 0, :]
print npzfile['emis_est'][0, 1, :]
print npzfile['elapsed_time'][0, 0]
print npzfile['elapsed_time'][0, 1]
