import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold


def data_preprocess(n_samples: int = 12):
    train_features = pd.read_csv('train_features.csv')
    train_labels = pd.read_csv('train_labels.csv')
    num_of_patient = train_features.pid.nunique()  # number of patients
    train_features_sel = data_sift(data=train_features)
    train_mat = train_features_sel.iloc[:, 2:].to_numpy()
    train_mat = train_mat.reshape(num_of_patient, n_samples, -1)
    pids = train_features.pid.to_numpy()
    # todo some reshape to the outputs?
    return train_mat, pids, train_labels.to_numpy()


def data_sift(data: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    threshold = 1 - threshold
    threshold *= data.shape[0]
    features_sel = data.dropna(axis=1, how='any', thresh=threshold)
    return features_sel


def imputer(mat: np.ndarray, pid_ind: int) -> np.ndarray:
    imp = KNNImputer(n_neighbors=5, weights="distance")
    x = imp.fit_transform(X=mat[pid_ind, :, :])
    return x


def task1():
    pass


def task2():
    pass


def task3():
    pass


if __name__ == '__main__':
    x_train, x_t_pid, y_train = data_preprocess(n_samples=12)
    # todo here the y_train contains all the labels. Perhaps need to change according to task id.
    pass
