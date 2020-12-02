#Downloads and prepares the data 
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
