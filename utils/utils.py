import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def convert_to_numpy(data):
    """ Converts data to a numpy array. """
    if isinstance(data, np.ndarray):  # already numpy
        return data        

    if isinstance(data, list):
        return np.array(data)
    
    if isinstance(data, pd.core.series.Series) or isinstance(data, pd.core.frame.DataFrame):
        return data.values
    
    raise TypeError("data is not a valid array type.")


def convert_and_expand(X):
    """ Converts data to numpy and reshapes if necessary. """
    X = convert_to_numpy(X)
    if X.ndim == 1:  # convert to 2D
        X = X[:,np.newaxis]
        
    return X

def prepare_data(Z, Y, X_cov):
    """ Standardizes data and split into train/test data  """
    Z = convert_to_numpy(Z)
    Y = convert_to_numpy(Y)
    X_cov = convert_to_numpy(X_cov)

    n, m = X_cov.shape
    train_idxs, test_idxs = train_test_split(np.arange(n), test_size=0.2, random_state=1)

    # numerical columns
    num_cols = [c for c in range(m) if len(np.unique(X_cov[:, c])) > 2] 


    X_train = X_cov[train_idxs,:]
    X_test = X_cov[test_idxs,:]

    Y_train = Y[train_idxs,:]
    Y_test= Y[test_idxs,:]

    Z_train = Z[train_idxs,:]
    Z_test= Z[test_idxs,:]

    # standardize data
    scaler_ = preprocessing.StandardScaler().fit(X_train[:,num_cols])
    X_train_scaled = np.copy(X_train)
    X_train_scaled[:,num_cols] = scaler_.transform(X_train[:,num_cols])

    X_test_scaled = np.copy(X_test)
    X_test_scaled[:,num_cols] = scaler_.transform(X_test[:,num_cols])


    return X_train_scaled, Y_train, Z_train, X_test_scaled, Y_test, Z_test
