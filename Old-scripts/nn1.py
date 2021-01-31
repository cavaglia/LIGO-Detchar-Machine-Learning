import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.neural_network import MLPClassifier

np.set_printoptions(threshold=np.inf)
fileName = 'L-L1_GWOSC_O2_16KHZ_R1-1181114368-4096.hdf5'
dataFile = h5py.File(fileName, 'r')
strain = dataFile['strain']['Strain'].value
X = strain
dqInfo = dataFile['quality']['simple']
bitnameList = dqInfo['DQShortnames'].value
nbits = len(bitnameList)
for bit in range(nbits):
    print(bit, bitnameList[bit])
qmask = dqInfo['DQmask'].value
print(len(qmask))
print(qmask)
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#clf.fit(X, y)

