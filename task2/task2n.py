import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
# from sklearn.svm import SVC, SVR
import sklearn.metrics as metrics

from tqdm import trange
import time


class Task2:
    def __init__(self):
        self.st1_labels_col = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
                               'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                               'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
        self.st2_labels_col = ['LABEL_Sepsis']
        self.st3_labels_col = ['LABEL_RRate',
                               'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
        self.st_labels_col = [self.st1_labels_col, self.st2_labels_col, self.st3_labels_col]
        self.x_train, self.x_t_pid = self.x_data_preprocess('task2/train_features.csv', n_samples=12)
        self.x_test, self.x_test_pid = self.x_data_preprocess('task2/test_features.csv', n_samples=12)
        self.y_st1, self.y_st2, self.y_st3 = self.y_data_preprocess('task2/train_labels.csv')
        self.y_st = [self.y_st1, self.y_st2, self.y_st3]
        
        self.df = pd.DataFrame({'pid': self.x_test_pid})
        
        self.st1_mdl, self.st2_mdl, self.st3_mdl = [], [], []
        self.st_mdl = [self.st1_mdl, self.st2_mdl, self.st3_mdl]
        
        self.paramsGrid_t = [{'n_estimators': [75, 100, 125, 150, 175]}, [{'n_estimators': [100, 125, 150, 175, 200]}], []]
        self.alg_t = [RandomForestClassifier, RandomForestClassifier, HistGradientBoostingRegressor]
        
        self.paramsGrid_h = [{'max_iter': [100, 150], 'max_leaf_nodes': [31, 50]}, {}, {}]
        self.alg_h = [HistGradientBoostingClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor]
        
        # self.alg_s = [SVC, SVC, SVR]
    
    def task(self, id, mode='fixed_param_train', model_group='t', **kwargs):
        """
        :param id: The subtask id. Can be chosen from (1, 2, 3).
        :param mode: Specify the operating mode for the training/cross-validation/prediction. Can be chosen from 
        ['param_grid_search', 'fixed_param_train', 'cross_validation', 'predict']
        :param model_group: Specify the group of model/algorithms. Currently available: 't' for random forest, 'h' for histGradientBoosting, 's' for SVM-related ones (Unavailable now)
        :param kwargs: necessary parameters for the fixed-param-train mode.
        """
        id -= 1
        assert id in {0, 1, 2} and isinstance(id, int)
        modes = ['param_grid_search', 'fixed_param_train', 'cross_validation', 'predict']
        assert mode in modes, ValueError("The mode is invalid!")
        assert model_group in ('t', 'h',)
        
        print(f"\n>>Task {id + 1}: {mode}<<\n")
        # task1 rf [150, 175, 175, 150, 150, 175, 150, 150, 175, 175]
        
        """The following part are for the algorithm groups."""
        
        if model_group == 't':
            """
            The implementation of RandomForest approaches. But this cannot be done for the regression task
            """
            if mode == 'param_grid_search':
                params = self.paramsGrid_t[id]
                self.st_mdl[id], best_params = self.__tGridSearch(id, params)
                print(best_params)
            elif mode == 'fixed_param_train':
                if 'num_estimators' in kwargs.keys():
                    self.__tFixedTrain(id, kwargs['num_estimators'])
                else:
                    self.__tFixedTrain(id)
            elif mode == 'cross_validation':
                if 'num_estimators' in kwargs.keys():
                    self.__tCV(id, kwargs['num_estimators'])
                else:
                    self.__tCV(id)
            elif mode == 'predict':
                if 'num_estimators' in kwargs.keys():
                    self.__Predict(id, kwargs['num_estimators'])
                else:
                    raise ValueError
        elif model_group == 'h':
            """
            The implementation of HistGradientBoosting approaches.
            """
            if mode == 'param_grid_search':
                params = self.paramsGrid_h[id]
                self.st_mdl[id], best_params = self.__hGridSearch(id, self.paramsGrid_h)
                print(best_params)
            elif mode == 'fixed_param_train':
                self.__hFixedTrain(id)
            elif mode == 'cross_validation':
                self.__hCV(id)
            elif mode == 'predict':
                self.__Predict(id)
    
    def output(self):
        self.df.to_csv('task2/prediction' + str(int(time.time()))[-5:] + '.csv', 
                       index=False, float_format='%.3f')
        self.df.to_csv('task2/prediction.zip', index=False, float_format='%.4f', compression='zip')
        
    def __tGridSearch(self, id, paramsDict):
        models = []
        best_params = {'n_estimators': []}
        for i in trange(len(self.st_labels_col[id])):
            clf = GridSearchCV(self.alg_t[id](), paramsDict, scoring='roc_auc', cv=5, n_jobs=-1)
            clf.fit(self.x_train, self.y_st[id][:, i])
            print(f"\nBest parameter for label {self.st_labels_col[id][i]} is {clf.best_params_}.\n")
            models.append(clf)
            best_params['n_estimators'].append(clf.best_params_['n_estimators'])
        return models, best_params
    
    def __tFixedTrain(self, id, num_estimators=[150]*10):
        assert len(num_estimators) == len(self.st_labels_col[id])
        for i, label in enumerate(self.st_labels_col[id]):
            if id in (0, 1):
                clf = self.alg_t[id](n_estimators=num_estimators[i])
                clf.fit(self.x_train, self.y_st[id][:, i])
                print(f"Label {label} Score: {metrics.roc_auc_score(self.y_st[id][:, i], clf.predict_proba(self.x_train)[:, 1])}.\n")
            else:
                clf = self.alg_t[id](max_depth=3)
                clf.fit(self.x_train, self.y_st[id][:, i])
                print(f"Label {label} Score: {metrics.r2_score(self.y_st[id][:, i], clf.predict(self.x_train))}.\n")
            self.st_mdl[id].append(clf)
            
    
    def __tCV(self, id, num_estimators=[150]*10):
        assert len(num_estimators) == len(self.st_labels_col[id])
        for i, label in enumerate(self.st_labels_col[id]):
            if id in (0, 1):
                clf = self.alg_t[id](n_estimators=num_estimators[i])
            else:
                clf = self.alg_t[id](max_depth=3)
            scores = cross_val_score(clf, self.x_train, self.y_st[id][:, i], cv=5, scoring='roc_auc')
            print("Label " + label + " Cross-validation score is {score:.3f},"
                  " standard deviation is {err:.3f}."
                  .format(score=scores.mean(), err=scores.std()))
    
    def __Predict(self, id):
        assert self.st_mdl[id] is not None
        for i, label in enumerate(self.st_labels_col[id]):
            clf = self.st_mdl[id][i]
            predictions = clf.predict_proba(self.x_test)[:, 1] if id in (0, 1) else clf.predict(self.x_test)
            self.df[label] = predictions
        print("--Prediction Finished!--")
    
    def __hGridSearch(self, id, paramsDict):
        models = []
        best_params = {'max_iter': [], 'max_leaf_nodes': [],}
        for i in trange(len(self.st_labels_col[id])):
            clf = GridSearchCV(self.alg_h[id](), paramsDict, scoring='roc_auc', cv=5, n_jobs=-1)
            clf.fit(self.x_train, self.y_st[id][:, i])
            print(f"\nBest parameter for label {self.st_labels_col[id][i]} is {clf.best_params_}.\n")
            models.append(clf)
            best_params['max_iter'].append(clf.best_params_['max_iter'])
            best_params['max_leaf_nodes'].append(clf.best_params_['max_leaf_nodes'])
        return models, best_params
    
    def __hFixedTrain(self, id):
        for i, label in enumerate(self.st_labels_col[id]):
            clf = self.alg_h[id]()
            clf.fit(self.x_train, self.y_st[id][:, i])
            if id in (0, 1):
                print(f"Label {label} Score: {metrics.roc_auc_score(self.y_st[id][:, i], clf.predict_proba(self.x_train)[:, 1])}.\n")
            else:
                print(f"Label {label} Score: {metrics.r2_score(self.y_st[id][:, i], clf.predict(self.x_train))}.\n")
            self.st_mdl[id].append(clf)
    
    def __hCV(self, id):
        for i, label in enumerate(self.st_labels_col[id]):
            clf = self.alg_h[id]()
            scores = cross_val_score(clf, self.x_train, self.y_st[id][:, i], cv=5, scoring='roc_auc')
            print("Label " + label + " Cross-validation score is {score:.3f},"
                  " standard deviation is {err:.3f}."
                  .format(score=scores.mean(), err=scores.std()))

    def x_data_preprocess(self, filename, n_samples: int = 12):
        x_features = pd.read_csv(filename)
        num_of_patient = x_features.pid.nunique()  # number of patients

        # original data
        x_mat = x_features.iloc[:, 2:].to_numpy()

        # expand features at different time into new features
        x_mat = np.swapaxes(x_mat.reshape(
            num_of_patient, n_samples, x_mat.shape[-1]), 1, 2).reshape(num_of_patient, -1)

        # standardize
        x_mat = self.standardize_mat(x_mat)
        
        pids = x_features.pid.to_numpy()
        pids, ind = np.unique(pids, return_index=True)
        pids = pids[np.argsort(ind)]
        return x_mat, pids

    def y_data_preprocess(self, filename):
        train_labels = pd.read_csv(filename)
        st1_gt, st2_gt, st3_gt = train_labels[self.st1_labels_col], train_labels[self.st2_labels_col], train_labels[self.st3_labels_col]
        return st1_gt.to_numpy(), st2_gt.to_numpy(), st3_gt.to_numpy()
    
    @staticmethod
    def standardize_mat(x):
        for col in range(x.shape[1]):
            x[np.where(np.isnan(x[:, col])), col] = np.nanmean(x[:, col])  # fill col mean values into the nans
            
            x[:, col] = (x[:, col] - np.mean(x[:, col])) / np.std(x[:, col])
        return x


if __name__ == '__main__':
    task2 = Task2()
    # task2.task(2, mode='param_grid_search')
    # task2.task(2, mode='param_grid_search', model_group='h')
    # task2.task(3, mode='fixed_param_train', model_group='h')
    # task2.task(3, mode='predict', model_group='h')
    # task2.task(2, mode='cross_validation', num_estimators=[150])
    # task2.task(1, mode='fixed_param_train', model_group='h')
    
    for i in (1, 2, 3):
        task2.task(i, mode='fixed_param_train', model_group='h')
        task2.task(i, mode='predict', model_group='h')
    # task2.output()