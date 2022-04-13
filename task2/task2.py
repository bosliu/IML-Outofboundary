import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import AdaBoostClassifier

from typing import List


def x_data_preprocess(n_samples: int = 12):
    train_features = pd.read_csv('train_features.csv')
    test_features = pd.read_csv('test_features.csv')
    num_of_patient = train_features.pid.nunique()  # number of patients
    train_mat = train_features.iloc[:, 2:].to_numpy()  # .reshape(num_of_patient, n_samples, -1)
    # as a numpy array, perform data selection
    kept_col = data_sift(data=train_mat, num_of_patient=num_of_patient, n_samples=n_samples, threshold=0.95)
    train_mat = train_mat[:, kept_col]
    test_mat = test_features.iloc[:, 2:].to_numpy()[:, kept_col]
    pids = train_features.pid.to_numpy()
    return train_mat, test_mat, pids


def data_sift(data: np.ndarray, num_of_patient: int, n_samples: int, threshold: float) -> List:
    """

    Select (i.e., delete) the columns based on two conditions:
        - If there exists one patient that all the samples in one column is nan;
        - And the entire column has more than threshold nans.
    ------
    Logics of coding: Mark the columns to be removed with the set difference between the first condition and the second.
    ------
    TODO: [!!!] The first condition removes almost all the columns! Maybe this should be deprecated?
    ------
    :param data: Input data, as a numpy array
    :param num_of_patient: number of patient
    :param n_samples: number of samples per patient.
    Product of n_samples with num_of_patient should be the shape[0] of data
    :param threshold: The threshold of keeping the data. E.g., if the lacking of a column inside a patient is greater
    than the threshold, the column would be deleted.
    :return: The column index to be kept inside the data.
    Returning the index since the same operation is to be performed on the test data as well.
    """
    assert num_of_patient * n_samples == data.shape[0], ValueError("The dimension of the data does not match!")
    # check and delete the columns with one patient missing all data (12 nan in one column)
    # threshold: missing less than threshold*data.shape[0] -> still keep
    patients = np.linspace(0, data.shape[0] - 1, num=num_of_patient).astype(int)
    kept_col = set([x for x in range(data.shape[1])])
    col_to_del = set()
    for p in patients:
        datablock = data[p: p + n_samples, :]
        for col in range(data.shape[1]):
            if col not in col_to_del and np.all(np.isnan(datablock[:, col])):
                col_to_del.add(col)
    still_keep_col = set(np.where(np.sum(np.isnan(data), axis=0) <= data.shape[0] * threshold)[0].tolist())
    kept_col = list(kept_col - (col_to_del - still_keep_col))
    return kept_col


def imputer(mat: np.ndarray, pid_ind: int, method: str = 'kNN') -> np.ndarray:
    if method == 'kNN':
        imp = KNNImputer(n_neighbors=5, weights="distance")
    elif method == 'Simple':
        imp = SimpleImputer(strategy='median')
    else:
        raise ValueError("Wrong imputer implemented!")
    x = imp.fit_transform(X=mat[pid_ind, :, :])
    return x


def y_preprocess():
    train_labels = pd.read_csv('train_labels.csv')
    st1_gt, st2_gt, st3_gt = train_labels[st1_labels_col], train_labels[st2_labels_col], train_labels[st3_labels_col]
    return st1_gt.to_numpy(), st2_gt.to_numpy(), st3_gt.to_numpy()


def task1(x: np.ndarray, y: np.ndarray):
    for i, label in enumerate(st1_labels_col):
        ppl = make_pipeline(
            SimpleImputer(strategy='median'),
            # Potential problem with KNN Imputer: if some sample lacks an entire column of data, it would be directly deleted -> dimension of the data may not match
            StandardScaler(),
            # todo some classifier should be here! perhaps try some kernelized SVM?
        )

    pass


def task2():
    pass


def task3():
    pass


if __name__ == '__main__':
    global st1_labels_col, st2_labels_col, st3_labels_col

    st1_labels_col = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
                      'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                      'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
    st2_labels_col = ['LABEL_Sepsis']
    st3_labels_col = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

    x_train, x_test, x_t_pid = x_data_preprocess(n_samples=12)
    y_st1, y_st2, y_st3 = y_preprocess()

    task1(x=x_train, y=y_st1)
