import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
import sklearn.metrics as metrics

from tqdm import trange

from typing import List


def x_data_preprocess(filename, n_samples: int = 12):
    x_features = pd.read_csv(filename)
    num_of_patient = x_features.pid.nunique()  # number of patients

    # original data
    train_mat = x_features.iloc[:, 2:].to_numpy()
    # test_mat = test_features.iloc[:, 2:].to_numpy()

    # expand features at different time into new features
    train_mat = np.swapaxes(train_mat.reshape(num_of_patient, n_samples, train_mat.shape[-1]), 1, 2).reshape(num_of_patient, -1)
    # train_mat = reshape_mat(train_mat, num_of_patient, n_samples)
    # test_mat = reshape_mat(test_mat, num_of_patient_test, n_samples)

    # standardize
    train_mat = standardize_mat(train_mat)
    # test_mat = standardize_mat(train_mat)
    # as a numpy array, perform data selection
    # kept_col = data_sift(
    #     data=train_mat, num_of_patient=num_of_patient, n_samples=n_samples, threshold=0.95)
    # train_mat = train_mat[:, kept_col]
    # test_mat = test_features.iloc[:, 2:].to_numpy()[:, kept_col]
    pids = x_features.pid.to_numpy()
    pids, ind = np.unique(pids, return_index=True)
    pids = pids[np.argsort(ind)]
    return train_mat, pids


# def reshape_mat(train_mat, num_of_patient, n_samples: int = 12):
#     h, w = train_mat.shape
#     train_arr = []
#     for i in range(num_of_patient):
#         for k in range(w):
#             for j in range(n_samples):
#                 train_arr.append(train_mat[i*n_samples+j, k])
#     train_mat = np.reshape(train_arr, (num_of_patient, n_samples*w))
#     return train_mat


def standardize_mat(train_mat):
    for col in range(train_mat.shape[1]):
        train_mat[np.where(np.isnan(train_mat[:, col])), col] = np.nanmean(train_mat[:, col])  # fill col mean values into the nans
        train_mat[:, col] = (train_mat[:, col] - np.mean(train_mat[:, col])) / np.std(train_mat[:, col])
    # np.savetxt("train_features_standardized.csv", train_mat, delimiter=",")
    return train_mat

"""
def imputer(mat: np.ndarray, pid_ind: int, method: str = 'kNN') -> np.ndarray:
    if method == 'kNN':
        imp = KNNImputer(n_neighbors=5, weights="distance")
    elif method == 'Simple':
        imp = SimpleImputer(strategy='median')
    else:
        raise ValueError("Wrong imputer implemented!")
    x = imp.fit_transform(X=mat[pid_ind, :, :])
    return x
"""


def y_preprocess():
    train_labels = pd.read_csv('task2/train_labels.csv')
    st1_gt, st2_gt, st3_gt = train_labels[st1_labels_col], train_labels[st2_labels_col], train_labels[st3_labels_col]
    return st1_gt.to_numpy(), st2_gt.to_numpy(), st3_gt.to_numpy()


def task1(x: np.ndarray, y: np.ndarray):
    params = {'n_estimators': [50, 75, 100, 125, 150]}
    models = []
    for i in trange(len(st1_labels_col)):
        # ppl = make_pipeline(
        #     # StandardScaler(),
        #     RandomForestClassifier()
        #     # AdaBoostClassifier(n_estimators=120)
        #     )
        # scores = cross_val_score(ppl, x, y[:, i],
        #                          cv=5,
        #                          scoring='roc_auc',
        #                          verbose=True)
        # print("Cross-validation score is {score:.3f},"
        #       " standard deviation is {err:.3f}"
        #       .format(score=scores.mean(), err=scores.std()))
        
        clf = GridSearchCV(RandomForestClassifier(), params, scoring='roc_auc', cv=5, n_jobs=-1)
        clf.fit(x, y[:, i])
        print(f"\nBest parameter for label {st1_labels_col[i]} is {clf.best_params_}.\n")
    models.append(clf)
    # return ppl
    return models


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
        ppl = make_pipeline(
            # SimpleImputer(strategy='median'),
            HistGradientBoostingRegressor(max_depth=3))
        scores = cross_val_score(ppl, x, y[:, i],
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
    x_train, x_t_pid = x_data_preprocess('task2/train_features.csv', n_samples=12)
    x_test, x_test_pid = x_data_preprocess('task2/test_features.csv', n_samples=12)
    y_st1, y_st2, y_st3 = y_preprocess()

    # output dataframe
    df = pd.DataFrame({'pid': x_test_pid})

    # Task 1
    print("Task 1")
    models = task1(x=x_train, y=y_st1)
    # for i, label in enumerate(st1_labels_col):
    #     ppl = ppl.fit(x_train, y_st1[:, i].ravel())
    #     print("Training score:", metrics.roc_auc_score(
    #         y_st1[:, i], ppl.predict_proba(x_train)[:, 1]))
    #     predictions = ppl.predict_proba(x_test)[:, 1]
    #     df[label] = predictions
    for i, label in enumerate(st1_labels_col):
        mdl = models[i]
        print(f"Label {label} has roc_auc score {metrics.roc_auc_score(y_st1[:, i], mdl.predict_proba(x_train)[:, 1])}.")
        # predictions = mdl.predict_proba(x_test)[:, 1]
        # df[label] = predictions
    """
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
    """
