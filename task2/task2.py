import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn.metrics as metrics
from sklearn.ensemble import HistGradientBoostingRegressor

from typing import List


def x_data_preprocess(filename, n_samples: int = 12):
    train_features = pd.read_csv(filename)
    # test_features = pd.read_csv('test_features.csv')
    num_of_patient = train_features.pid.nunique()  # number of patients
    # num_of_patient_test = test_features.pid.nunique()
    # .reshape(num_of_patient, n_samples, -1)

    # original data
    train_mat = train_features.iloc[:, 2:].to_numpy()
    # test_mat = test_features.iloc[:, 2:].to_numpy()

    # expand features at different time into new features
    train_mat = reshape_mat(train_mat, num_of_patient, n_samples)
    # test_mat = reshape_mat(test_mat, num_of_patient_test, n_samples)

    # standardize
    train_mat = standardize_mat(train_mat)
    # test_mat = standardize_mat(train_mat)
    # as a numpy array, perform data selection
    # kept_col = data_sift(
    #     data=train_mat, num_of_patient=num_of_patient, n_samples=n_samples, threshold=0.95)
    # train_mat = train_mat[:, kept_col]
    # test_mat = test_features.iloc[:, 2:].to_numpy()[:, kept_col]
    pids = train_features.pid.to_numpy()
    pids, ind = np.unique(pids, return_index=True)
    pids = pids[np.argsort(ind)]
    return train_mat, pids


# def reshape_mat(train_mat, num_of_patient, n_samples: int = 12):
#     h, w = train_mat.shape
#     A = np.zeros((1, n_samples*w))
#     for i in range(num_of_patient):
#         sub_train_mat = train_mat[n_samples*i:n_samples*(i+1)]
#         b = np.reshape(sub_train_mat, n_samples*w, order='F')
#         A = np.append(A, [b], axis=0)
#     np.savetxt("train_features_editted.csv", A, delimiter=",")
#     return A


def reshape_mat(train_mat, num_of_patient, n_samples: int = 12):
    h, w = train_mat.shape
    train_arr = []
    for i in range(num_of_patient):
        for k in range(w):
            for j in range(n_samples):
                train_arr.append(train_mat[i*n_samples+j, k])
    train_mat = np.reshape(train_arr, (num_of_patient, n_samples*w))
    return train_mat


def standardize_mat(train_mat):
    for col in range(train_mat.shape[1]):
        train_mat[np.where(np.isnan(train_mat[:, col])),
                  col] = np.nanmean(train_mat[:, col])
        train_mat[:, col] = (
            train_mat[:, col] - np.mean(train_mat[:, col])) / np.std(train_mat[:, col])
    # np.savetxt("train_features_standardized.csv", train_mat, delimiter=",")
    return train_mat


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
    assert num_of_patient * n_samples == data.shape[0], ValueError(
        "The dimension of the data does not match!")
    # check and delete the columns with one patient missing all data (12 nan in one column)
    # threshold: missing less than threshold*data.shape[0] -> still keep
    patients = np.linspace(
        0, data.shape[0] - 1, num=num_of_patient).astype(int)
    kept_col = set([x for x in range(data.shape[1])])
    col_to_del = set()
    for p in patients:
        datablock = data[p: p + n_samples, :]
        for col in range(data.shape[1]):
            if col not in col_to_del and np.all(np.isnan(datablock[:, col])):
                col_to_del.add(col)
    still_keep_col = set(np.where(
        np.sum(np.isnan(data), axis=0) <= data.shape[0] * threshold)[0].tolist())
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
            # SimpleImputer(strategy='median'),
            # Potential problem with KNN Imputer: if some sample lacks an entire column of data, it would be directly deleted -> dimension of the data may not match
            StandardScaler(),
            # todo some classifier should be here! perhaps try some kernelized SVM?
            HistGradientBoostingClassifier())
        scores = cross_val_score(ppl, x, y[:, i],
                                 cv=5,
                                 scoring='roc_auc',
                                 verbose=True)
        print("Cross-validation score is {score:.3f},"
              " standard deviation is {err:.3f}"
              .format(score=scores.mean(), err=scores.std()))
    return ppl


def task2(x: np.ndarray, y: np.ndarray):
    ppl = make_pipeline(
        # SimpleImputer(strategy='median'),
        StandardScaler(),
        HistGradientBoostingClassifier())

    scores = cross_val_score(ppl, x, y.ravel(),
                             cv=5,
                             scoring='roc_auc',
                             verbose=True)
    print("Cross-validation score is {score:.3f},"
          " standard deviation is {err:.3f}"
          .format(score=scores.mean(), err=scores.std()))
    return ppl


def task3(x: np.ndarray, y: np.ndarray):
    for i, label in enumerate(st3_labels_col):
        pipeline = make_pipeline(
            # SimpleImputer(strategy='median'),
            HistGradientBoostingRegressor(max_depth=3))
        scores = cross_val_score(pipeline, x, y[:, i],
                                 cv=5,
                                 scoring='r2',
                                 verbose=True)
        print("Cross-validation score is {score:.3f},"
              " standard deviation is {err:.3f}"
              .format(score=scores.mean(), err=scores.std()))
    return ppl


if __name__ == '__main__':
    global st1_labels_col, st2_labels_col, st3_labels_col

    st1_labels_col = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
                      'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                      'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
    st2_labels_col = ['LABEL_Sepsis']
    st3_labels_col = ['LABEL_RRate', 'LABEL_ABPm',
                      'LABEL_SpO2', 'LABEL_Heartrate']

    # data preprocess
    x_train, x_t_pid = x_data_preprocess('train_features.csv', n_samples=12)
    x_test, x_test_pid = x_data_preprocess('test_features.csv', n_samples=12)
    y_st1, y_st2, y_st3 = y_preprocess()

    # output dataframe
    df = pd.DataFrame({'pid': x_test_pid})

    # Task 1
    print("Task 1")
    ppl = task1(x=x_train, y=y_st1)
    for i, label in enumerate(st1_labels_col):
        ppl = ppl.fit(x_train, y_st1[:, i].ravel())
        print("Training score:", metrics.roc_auc_score(
            y_st1[:, i], ppl.predict_proba(x_train)[:, 1]))
        predictions = ppl.predict_proba(x_test)[:, 1]
        df[label] = predictions

    # Task 2
    print("Task 2")
    ppl2 = task2(x=x_train, y=y_st2)
    ppl2 = ppl2.fit(x_train, y_st2.ravel())
    predictions = ppl2.predict_proba(x_test)[:, 1]
    print("Training score:", metrics.roc_auc_score(
        y_st2, ppl2.predict_proba(x_train)[:, 1]))
    df[st2_labels_col[0]] = predictions

    # Task 3
    print("Task 3")
    ppl3 = task3(x=x_train, y=y_st3)
    for i, label in enumerate(st3_labels_col):
        pipeline = make_pipeline(
            # SimpleImputer(strategy='median'),
            StandardScaler(),
            HistGradientBoostingRegressor(max_depth=3))
        pipeline = pipeline.fit(x_train, y_st3[:, i])
        predictions = pipeline.predict(x_test)
        print("Training score:", metrics.r2_score(
            y_st3[:, i], pipeline.predict(x_train)))
        df[label] = predictions

    print(df)
    df.to_csv('prediction.csv', index=False, float_format='%.4f')
