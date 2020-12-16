## Usage: python3.8 marco-devel.py --rate 4096 [--verbose] --start 0 --end 4096 [--standardize] --datafile H-H1_GWOSC_O2_4KHZ_R1-1181155328-4096.hdf5
## Data files must be in "Data subdirectory". Results go in "Results" subdirectory.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='count', help='Verbose mode')
parser.add_argument('--datafile', help='hdf data file from GWOSC. For example H-H1_GWOSC_O2_4KHZ_R1-1181155328-4096.hdf5', required=True)
parser.add_argument('--start',  type=int, help='The start time (in seconds) of data to analyze. (Positive integer, default = 0)', default=0, required=False)
parser.add_argument('--end',  type=int, help='The start time (in seconds) of data to analyze. (Positive integer, default = 4096)', default=4096, required=False)
parser.add_argument('--rate', type=int, help='Sampling rate (positive even integer. It must match the sampling rate of the LIGO data. Default = 4096)', default=4096, required=False)
parser.add_argument('--standardize', action='count', help='Standardize data')
args = parser.parse_args()

data_file = args.datafile
verbose = args.verbose
sampling_rate = args.rate
start_time = args.start
end_time = args.end
sampling_rate = args.rate
standardize = args.standardize

#np.set_printoptions(threshold=np.inf)

def read_DQ(data_file):
    currentDirectory = os.getcwd()
    dataFile = h5py.File(currentDirectory+'/Data/'+data_file, 'r')
    dqInfo = dataFile['quality']['simple']
    bitnameList = dqInfo['DQShortnames'][()]
    nbits = len(bitnameList)
    qmask = dqInfo['DQmask'][()] #0 b'DATA' #1 b'CBC_CAT1' #2 b'CBC_CAT2' #3 b'CBC_CAT3' #4 b'BURST_CAT1' #5 b'BURST_CAT2' #6 b'BURST_CAT3'
    Data = (qmask >> 0) & 1	 
    CBC_CAT1 = (qmask >> 1) & 1    #BURST_CAT1 = (qmask >> 4) & 1   
    CBC_CAT2 = (qmask >> 2) & 1    #BURST_CAT2 = (qmask >> 5) & 1
    CBC_CAT3 = (qmask >> 3) & 1    #BURST_CAT3 = (qmask >> 6) & 1

    CBC_CAT1Data = Data & CBC_CAT1    #BURST_CAT1Data = Data & BURST_CAT1
    CBC_CAT2Data = Data & CBC_CAT2    #BURST_CAT2Data = Data & BURST_CAT2
    CBC_CAT3Data = Data & CBC_CAT3    #BURST_CAT3Data = Data & BURST_CAT3

    CBC_allCATData = CBC_CAT1Data + CBC_CAT2Data + CBC_CAT3Data
    dataFile.close()

    return CBC_allCATData

def read_strain(sampling_rate,start_time,data_download):
    end_time = start_time + 1
    data_start = sampling_rate * start_time
    data_end = sampling_rate * end_time
    time_stamps =np.arange(data_start, data_end)
    data = pd.DataFrame({'Time':time_stamps,'Strain':data_download[data_start:data_end]})
    return data

def data_download(data_file):
    currentDirectory = os.getcwd()
    dataFile = h5py.File(currentDirectory+'/Data/'+data_file, 'r')
    data_download = np.array(dataFile['strain/Strain'][...])
    dataFile.close()
    return data_download

#This function standardizes the data so that the mean is zero and the standard deviation is 1. Requires the dataframe made with real_data and returns the
#standrdized strain
def standardize_data(X):
    data_reshaped = np.asarray(X).reshape(-1, 1)
    scaler = StandardScaler().fit(data_reshaped)
    ####print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, np.sqrt(scaler.var_)))
    normalized = scaler.transform(data_reshaped)
    ####inversed = scaler.inverse_transform(normalized)   
    return normalized

def build_training_dataset(sampling_rate,start_time,end_time,data_download,DQ):
    labeled_data = pd.DataFrame(columns=['Strain', 'Label'])
    i = start_time
    while i < end_time:
        data = read_strain(sampling_rate,i,data_download)
        if np.isnan(data.Strain[0]):   #Alternative, possibly slower: if data.isnull().values.any():
            if verbose:
                print('Data starting at %d second(s) are not defined. Skipping.' % (i))
            i+=1 
            continue
        X = data['Strain'].values
        if standardize:
        	X = standardize_data(X).reshape(1, -1)[0]
        labeled_data=labeled_data.append({'Time':i,'Strain':X,'Label':DQ[i]}, ignore_index=True)
        i+=1
    return labeled_data

def build_training_model(dataset):
    #clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (10, 4), random_state = 1)
    clf = MLPClassifier()
    clf.fit(dataset['Strain'].to_list(), dataset['Label'].to_list())
    return clf

def save_predicted_labels(data_file,input_dataset,model):
    currentDirectory = os.getcwd()
    [filename_body, filename_ext] = data_file.split('.')
    filename = currentDirectory+'/Results/'+filename_body+'.txt'
    if os.path.isfile(filename):
        os.remove(filename)
    labels = model.predict(input_dataset['Strain'].to_list())
    labels_index = input_dataset['Time'].astype('int').to_list()
    predicted_labels = pd.DataFrame({'Time':labels_index,'Label':labels})
    with open(filename, 'a') as f:
        f.write('# Predicted labels for ' + data_file + '\n')
        predicted_labels.to_csv(f,sep='\t',index=False)
    return predicted_labels

def compute_F1_score(y_true,y_pred)
    predicted_labels = y_pred['labels']
    score = f1_score(y_true, predicted_labels)
return score

#--- Body of program

if verbose:
    print('Starting!')
    print('Data file: %s, start time: %d s, end time: %d s, sampling rate: %d Hz' % (data_file, start_time, end_time, sampling_rate))

data = data_download(data_file)
DQ = read_DQ(data_file)
training_dataset = build_training_dataset(sampling_rate,start_time,end_time,data,DQ)
trained_model = build_training_model(training_dataset)
DQ_predicted = save_predicted_labels(data_file,training_dataset,trained_model)

## Add here your F1 score compute_F1(DQ,DQ_predicted)
## F1_score = compute_F1_score(DQ,DQ_predicted)

sys.exit()

        #plt.figure(0)
        #plt.plot(strain, label='Available data',color='b')
        #plt.legend(loc=1)
        #plt.xlabel('Time (s)')
        #plt.savefig('data.png')
        #plt.figure(1)
        #plt.plot(CBC_CAT1Data, label='Available data passing CBC CAT1',color='r')
        #plt.plot(CBC_CAT2Data, label='Available data passing CBC CAT2',color='c')
        #plt.plot(CBC_CAT3Data, label='Available data passing CBC CAT3',color='g')
        ##plt.plot(BURST_CAT1Data, label='Available data passing BURST CAT1',color='r')
        ##plt.plot(BURST_CAT2Data, label='Available data passing BURST CAT2',color='c')
        ##plt.plot(BURST_CAT3Data, label='Available data passing BURST CAT3',color='g')
        #plt.legend(loc=1)
        #plt.xlabel('Time (s)')
        #plt.savefig('dq.png')

