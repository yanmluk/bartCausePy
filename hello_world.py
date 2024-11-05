from bartCausePy.r_to_py.r_to_py import init_R
from bartCausePy.bartCause.bart_cause import BARTCause
from bartCausePy.utils.utils import prepare_data

import pandas as pd
import numpy as np


if __name__ == '__main__':
    # init R
    init_R()
    # load data
    print("preparing data...")
    df = pd.read_csv('data.csv')
    # split data into train/test set
    X_train_scaled, y_train, Z_train, X_test_scaled, y_test, Z_test = prepare_data(df[['T']], df[['Y']], df.iloc[:,:6])

    # train BART model
    print("fitting BART model...")
    bart_eval = BARTCause()
    bart_eval.fit(X_train_scaled, y_train, Z_train, n_samples=1000,  n_burn=200,  n_chains=5)

    # predict ITE on test data
    print("predicting test data...")
    newData = np.concatenate((X_test_scaled, Z_test), axis=1)
    predicted_Z1, _, _ = bart_eval.predict(newData, infer_type="mu.1")
    predicted_Z0, _, _ = bart_eval.predict(newData, infer_type="mu.0")

    # print average ITE
    mean_law0 = predicted_Z0.mean()
    mean_law1 = predicted_Z1.mean()
    avg_ite = (predicted_Z1 - predicted_Z0).mean()
    std_ite = (predicted_Z1 - predicted_Z0).std()
    print("Avg ITE:", avg_ite,"Stdev ITE:",std_ite)