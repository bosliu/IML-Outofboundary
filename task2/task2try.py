import numpy as np
import pandas as pd


"""
DO NOT SUBMIT THIS FILE!
Playground with the code from the GitHub.
"""


def calculate_time_features(data, n_samples):
    x = []
    features = [np.nanmedian, np.nanmean, np.nanvar, np.nanmin,
           np.nanmax]
    for index in range(int(data.shape[0] / n_samples)):
        assert data[n_samples * index, 0] == data[n_samples * (index + 1) - 1, 0], \
        'Ids are {}, {}'.format(data[n_samples * index, 0], data[n_samples * (index + 1) - 1, 0])
        patient_data = data[n_samples * index:n_samples * (index + 1), 2:]
        feature_values = np.empty((len(features), data[:, 2:].shape[1]))
        for i, feature in enumerate(features):
            feature_values[i] = feature(patient_data, axis=0)
        x.append(feature_values.ravel())
    return np.array(x)


train_data = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_labels.csv')
test_data = pd.read_csv('test_features.csv')
x_train = calculate_time_features(train_data.to_numpy(), 12)
x_test = calculate_time_features(test_data.to_numpy(), 12)
