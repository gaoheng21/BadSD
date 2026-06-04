import numpy as np
loaded_weights = np.load('/home/server/science/gh/BadSD/gmm_test/final_weights_SST-2_BadNets_20260429_100825.npy').tolist()
print(max(loaded_weights))  