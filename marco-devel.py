## Usage: python3.8 marco-devel.py [--rate 4096] [--verbose] [--start_train POSITIVE INTEGER ] [--end_train POSITIVE INTEGER] [--standardize] [--whiten] [--filterfreq low,high] --trainfile trainfile.txt [--start_test POSITIVE INTEGER ] [--end_test POSITIVE INTEGER] --testfile testfile.txt
## GWOSC data files must be in "Data subdirectory". Datafile.txt must be in the dame directory of this script.
##    It must contain a list of GWOSC files to be analyzed , one per line. Results go in "Results" subdirectory.
import numpy as np
import os
import sys
import pandas as pd
import argparse
import h5py
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as metrics
from sklearn import preprocessing

from scipy import signal
from scipy.interpolate import interp1d

import matplotlib.mlab as mlab

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='count', help='Verbose mode')
parser.add_argument('--trainfile', help='Ascii file with list of hdf data file from GWOSC for training. Each line (example): H-H1_GWOSC_O2_4KHZ_R1-1181155328-4096.hdf5', required=True)
parser.add_argument('--testfile', help='Ascii file with list of hdf data file from GWOSC for testing. Each line (example): H-H1_GWOSC_O2_4KHZ_R1-1181155328-4096.hdf5', required=True)
parser.add_argument('--start_train',  type=int, help='The start time (in seconds) of training data to analyze. For debugging purposes. (Positive integer, default = 0)', default=0, required=False)
parser.add_argument('--end_train',  type=int, help='The end time (in seconds) of training data to analyze. For debugging purposes. (Positive integer, default = all the available time)', required=False)
parser.add_argument('--start_test',  type=int, help='The start time (in seconds) of testing data to analyze. For debugging purposes. (Positive integer, default = 0)', default=0, required=False)
parser.add_argument('--end_test',  type=int, help='The end time (in seconds) of testing data to analyze. For debugging purposes. (Positive integer, default = all the available time)', required=False)
parser.add_argument('--rate', type=int, help='Sampling rate (positive even integer. It must match the sampling rate of the LIGO data. Default = 4096)', default=4096, required=False)
parser.add_argument('--filterfreq', nargs='*', default=[], help="Filter frequencies: --filterfreq low,high ...")
parser.add_argument('--standardize', action='count', help='Standardize data')
parser.add_argument('--whiten', action='count', help='Whitens the data')
args = parser.parse_args()

train_file = args.trainfile
test_file = args.testfile
verbose = args.verbose
sampling_rate = args.rate
start_train_time = args.start_train
end_train_time = args.end_train
start_test_time = args.start_test
end_test_time = args.end_test
sampling_rate = args.rate
standardize = args.standardize
whiten = args.whiten

# ---------------------------

def read_strain(sampling_rate,start_time,data_download):
    end_time = start_time + 1
    data_start = sampling_rate * start_time
    data_end = sampling_rate * end_time
    time_stamps =np.arange(data_start, data_end)
    data = pd.DataFrame({'Time':time_stamps,'Strain':data_download[data_start:data_end]})
    return data

def data_download(data_file):
    currentDirectory = os.getcwd()
    listfile = currentDirectory + '/Data/' + data_file
    datafiles = pd.read_csv(currentDirectory + '/' + data_file,comment ='#',header=None)
    data_download = []
    CBC_allCATData = [] 
    for datafile in datafiles[0]:
        datafile = str(datafile)
        if verbose:
            print('Reading the data file %s sampled at %d Hz' % (datafile, sampling_rate))
        dataFile = h5py.File(currentDirectory+'/Data/'+ datafile, 'r')
        data_download = np.append(data_download,np.array(dataFile['strain/Strain'][...]))        
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

        CBC_allCATData = np.append(CBC_allCATData,CBC_CAT1Data + CBC_CAT2Data + CBC_CAT3Data)
       
        dataFile.close()

    return data_download, CBC_allCATData

def condition_data(fs,fband,whiten,strain,DQ):
    if len(fband) and whiten and verbose:
        print('Filtering the data between %d Hz and %d Hz and whitening the data...' %(fband[0],fband[1]))    
    elif len(fband) and verbose:
        print('Filtering the data between %d Hz and %d Hz...' %(fband[0],fband[1]))
    elif whiten and verbose:
       print('Whitening the data...')
    if len(fband):
        bb, ab = signal.butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
        normalization = np.sqrt((fband[1]-fband[0])/(fs/2))       
    notnans = np.nonzero(~np.isnan(strain))[0]
    notnans_arrays = np.split(notnans, np.where(np.diff(notnans) != 1)[0]+1)
    strain_conditioned = np.empty(0,dtype=int)
    for time in notnans_arrays:
        if len(fband):
            strain_filtered_time = signal.filtfilt(bb, ab, strain[time]) / normalization
        else:
            strain_filtered_time = strain[time]
        if whiten:
            Pxx, freqs = mlab.psd(strain_filtered_time, Fs = fs, NFFT = fs)
            psd = interp1d(freqs, Pxx)
            strain_conditioned_time = whiten_data(strain_filtered_time,psd,1/fs) 
        else:
            strain_conditioned_time = strain_filtered_time            
        strain_conditioned = np.append(strain_conditioned,strain_conditioned_time)    
    DQ_conditioned = DQ[np.nonzero(DQ)]
    return strain_conditioned, DQ_conditioned

def whiten_data(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strain)
    white_hf = hf/(np.sqrt(interp_psd(freqs)/dt/2.))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

#This function standardizes the data so that the mean is zero and the standard deviation is 1. Requires the dataframe made with real_data and returns the
#standardized strain
def standardize_data(X):
    data_reshaped = np.asarray(X).reshape(-1, 1)
    scaler = StandardScaler().fit(data_reshaped)
    normalized = scaler.transform(data_reshaped)
    return normalized

def build_dataset(sampling_rate,start_time,end_time,data_download,DQ):
    if verbose:
        print('Building a %d second-long dataset...' % (end_time - start_time))    
    labeled_data = pd.DataFrame(columns=['Strain', 'Label'])
    i = start_time
    while i < end_time:
        data = read_strain(sampling_rate,i,data_download)
        X = data['Strain'].values
        if standardize:
        	X = standardize_data(X).reshape(1, -1)[0]
        labeled_data=labeled_data.append({'Time':i,'Strain':X,'Label':DQ[i]}, ignore_index=True)
        i+=1
    data_length = len(labeled_data)
    if not data_length:
        print('There is no data to train or test. Aborting!')
        sys.exit()
    elif data_length < (end_time - start_time):
        print('Warning: Some data is not defined. The duration of the dataset is only %d second(s).' % (len(labeled_data)))    
        
    return labeled_data

def build_training_model(dataset):
    if verbose:
        print('Training the model...')    
   #clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (10, 4), random_state = 1)
    clf = MLPClassifier()
    clf.fit(dataset['Strain'].to_list(), dataset['Label'].to_list())
    return clf

def save_predicted_labels(data_file,input_dataset,model):
    if verbose:
        print('Testing the model...')    
    currentDirectory = os.getcwd()
    [filename_body, filename_ext] = data_file.split('.')
    filename = currentDirectory+'/Results/'+filename_body+'-prediction.txt'
    if os.path.isfile(filename):
        os.remove(filename)
    labels = model.predict(input_dataset['Strain'].to_list())
    labels_index = input_dataset['Time'].to_list()
    predicted_labels = pd.DataFrame({'Time':labels_index,'Label':labels},dtype=int)
    with open(filename, 'a') as f:
        f.write('# Predicted labels for ' + data_file + '\n')
        predicted_labels.to_csv(f,sep='\t',index=False)
    if verbose:
        print('Predicted labels are saved in ./Results/%s.' % (filename_body+'-prediction.txt'))    
    return predicted_labels

def save_true_labels(data_file,input_dataset):
    if verbose:
        print('Saving the true labels (for debugging)...')    
    currentDirectory = os.getcwd()
    [filename_body, filename_ext] = data_file.split('.')
    filename = currentDirectory+'/Results/'+filename_body+'-true.txt'
    if os.path.isfile(filename):
        os.remove(filename)
    labels = input_dataset['Label'].to_list()
    labels_index = input_dataset['Time'].to_list()
    true_labels = pd.DataFrame({'Time':labels_index,'Label':labels},dtype=int)
    with open(filename, 'a') as f:
        f.write('# True labels for ' + data_file + '\n')
        true_labels.to_csv(f,sep='\t',index=False)
    if verbose:
        print('True labels are saved in ./Results/%s.' % (filename_body+'-true.txt'))    
    return true_labels

def compute_metrics(data_file,training_set,predicted_labels):
    if verbose:
        print('Calculating the prediction metrics...')    
    currentDirectory = os.getcwd()
    [filename_body, filename_ext] = data_file.split('.')
    filename = currentDirectory+'/Results/'+filename_body+'-metrics.txt'
    if os.path.isfile(filename):
        os.remove(filename)
    y_true = training_set['Label'].to_list()
    y_pred = predicted_labels['Label'].to_list()
    predicted_metrics = metrics.classification_report(y_true, y_pred, zero_division='warn')
    cm_array = metrics.confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(cm_array) 
    with open(filename, 'a') as f:
        f.write('# Predicted labels for ' + data_file + ':\n\n')
        f.write(predicted_metrics)
        f.write('\n# Confusion matrix for ' + data_file + ':\n\n')
        cm.to_csv(f,sep='\t',index=True)
    if verbose:
        print('Prediction metrics are saved in ./Results/%s.' % (filename_body+'-metrics.txt'))    
    return

#--- Body of program

if verbose:
    print('Starting!')

if args.filterfreq:
    filter_freq = np.array(args.filterfreq[0].split(',')).astype(float)
else:
    filter_freq = []

# Reads and conditions the data for training

strain_train, DQ_train = data_download(train_file)
strain_train_conditioned, DQ_train_conditioned = condition_data(sampling_rate,filter_freq,whiten,strain_train,DQ_train)

if not end_train_time or end_train_time > len(DQ_train_conditioned):
   end_train_time = len(DQ_train_conditioned)
   print('Warning: The end time of the training set is %d second(s).' % (end_train_time))

# Builds the training dataset

training_dataset = build_dataset(sampling_rate,start_train_time,end_train_time,strain_train_conditioned,DQ_train_conditioned)

# Saves the true training labels (for debugging)

save_true_labels(train_file,training_dataset)

# Builds the model on the training data set

trained_model = build_training_model(training_dataset)

# Does the prediction on the same training dataset and saves the results (this step for debugging purposes) 

DQ_train_predicted = save_predicted_labels(train_file,training_dataset,trained_model)
compute_metrics(train_file,training_dataset,DQ_train_predicted)

# Reads and conditions the data for training

strain_test, DQ_test = data_download(test_file)
strain_test_conditioned, DQ_test_conditioned = condition_data(sampling_rate,filter_freq,whiten,strain_test,DQ_test)

if not end_test_time or end_test_time > len(DQ_test_conditioned):
   end_test_time = len(DQ_test_conditioned)
   print('Warning: The end time of the testing set is %d second(s).' % (end_test_time))

# Builds the training dataset

testing_dataset = build_dataset(sampling_rate,start_test_time,end_test_time,strain_test_conditioned,DQ_test_conditioned)

# Saves the true testing labels (for debugging)

save_true_labels(test_file,testing_dataset)

# Does the prediction on the same testing dataset using the trained model and saves the results 

DQ_test_predicted = save_predicted_labels(test_file,testing_dataset,trained_model)

compute_metrics(test_file,testing_dataset,DQ_test_predicted)

if verbose:
    print('Done!')

sys.exit()

