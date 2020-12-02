#Downloads and prepares the data.
#Usage: Put the GWOSC data file [data_file] in a Data directory. This reads [data_length] seconds of data in the file starting from start_time (in seconds).
#For example: real_data(4096,10,1,H-H1_GWOSC_O2_4KHZ_R1-1181642752-4096.hdf5) reads 1 second of data starting from second 10 in the file
#H-H1_GWOSC_O2_4KHZ_R1-1181642752-4096.hdf5. This function can be used to loop over the file and separate the data in chunks of data_length seconds.
#It returns a pandas dataframe with the time stamps and the strain values.
def real_data(sampling_rate,start_time,data_length,data_file):
    end_time = start_time + data_length
    data_start = sampling_rate * start_time
    data_end = sampling_rate * end_time
    currentDirectory = os.getcwd()
    dataFile = h5py.File(currentDirectory+'/Data/'+data_file, 'r')
    data_download = np.array(dataFile['strain/Strain'][...])
    dataFile.close()
    time_stamps =np.arange(data_start, data_end)
    real_data = pd.DataFrame({'time':time_stamps,'value':data_download[data_start:data_end]})
    return real_data
