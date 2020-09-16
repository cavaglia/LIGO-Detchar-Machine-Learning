import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import sys
import pandas as pd
import argparse
import h5py
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--datafile', help='Data file', required=True)
args = parser.parse_args()

filename = args.datafile

np.set_printoptions(threshold=np.inf)

dataFile = h5py.File(filename, 'r')

#keys = []
#for key in datafile.keys():
#	keys.append(key)
#
#print(keys)
#
#group = datafile[keys[1]]
#print(group.keys())

strain = dataFile['strain']['Strain'][()]
X = strain
dqInfo = dataFile['quality']['simple']
bitnameList = dqInfo['DQShortnames'][()]
nbits = len(bitnameList)

qmask = dqInfo['DQmask'][()]

#for bit in range(nbits):
#	print(bit, bitnameList[bit])

#0 b'DATA'
#1 b'CBC_CAT1'
#2 b'CBC_CAT2'
#3 b'CBC_CAT3'
#4 b'BURST_CAT1'
#5 b'BURST_CAT2'
#6 b'BURST_CAT3'

Data = (qmask >> 0) & 1
CBC_CAT1 = (qmask >> 1) & 1
CBC_CAT2 = (qmask >> 2) & 1
CBC_CAT3 = (qmask >> 3) & 1
#BURST_CAT1 = (qmask >> 4) & 1
#BURST_CAT2 = (qmask >> 5) & 1
#BURST_CAT3 = (qmask >> 6) & 1

CBC_CAT1Data = Data & CBC_CAT1
CBC_CAT2Data = Data & CBC_CAT2
CBC_CAT3Data = Data & CBC_CAT3
#BURST_CAT1Data = Data & BURST_CAT1
#BURST_CAT2Data = Data & BURST_CAT2
#BURST_CAT3Data = Data & BURST_CAT3

#plt.plot(Data, label='Available data',color='b')
plt.plot(CBC_CAT1Data, label='Available data passing CBC CAT1',color='r')
plt.plot(CBC_CAT2Data, label='Available data passing CBC CAT2',color='c')
plt.plot(CBC_CAT3Data, label='Available data passing CBC CAT3',color='g')
#plt.plot(BURST_CAT1Data, label='Available data passing BURST CAT1',color='r')
#plt.plot(BURST_CAT2Data, label='Available data passing BURST CAT2',color='c')
#plt.plot(BURST_CAT3Data, label='Available data passing BURST CAT3',color='g')
plt.legend(loc=1)
plt.xlabel('Time (s)')

plt.savefig('dq.png')

#print(len(qmask))
#print(qmask)

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#clf.fit(X, y)

sys.exit()


