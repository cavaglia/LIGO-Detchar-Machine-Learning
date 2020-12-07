#This function standardizes the data so that the mean is zero and the standard deviation is 1. Requires the dataframe made with real_data and returns the
#standrdized strain
def standardize(data):
    data_reshaped = np.asarray(data['value']).reshape((len(data['time']), 1))
    scaler = StandardScaler().fit(data_reshaped)
    #print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, np.sqrt(scaler.var_)))
    normalized = scaler.transform(data_reshaped)
    #inversed = scaler.inverse_transform(normalized)
    return normalized
