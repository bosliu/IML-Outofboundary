import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
import sklearn.metrics as metrics

from tqdm import trange


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
        self.paramsGrid = [{'n_estimators': [75, 100, 125, 150, 175]}, [{'n_estimators': [100, 125, 150, 175, 200]}], []]
        
        self.alg = [RandomForestClassifier, RandomForestClassifier, HistGradientBoostingRegressor(max_depth=3)]
    
    def task(self, id, mode='param_grid_search', **kwargs):
        id -= 1
        assert id in {0, 1, 2} and isinstance(id, int)
        modes = ['param_grid_search', 'fixed_param_train', 'cross_validation', 'predict']
        assert mode in modes, ValueError("The mode is invalid!")
        print(f"\n>>Task {id + 1}<<\n")
        # task1 [150, 175, 175, 150, 150, 175, 150, 150, 175, 175]
        if mode == 'param_grid_search':
            params = self.paramsGrid[id]
            self.st_mdl[id], _ = self.__tGridSearch(id, params)
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
                self.__tPredict(id, kwargs['num_estimators'])
            else:
                raise ValueError
        
    def __tGridSearch(self, id, paramsDict):
        models = []
        best_params = {'n_estimators': []}
        for i in trange(len(self.st_labels_col[id])):
            clf = GridSearchCV(RandomForestClassifier(), paramsDict, scoring='roc_auc', cv=5, n_jobs=-1)
            clf.fit(self.x_train, self.y_st[id][:, i])
            print(f"\nBest parameter for label {self.st_labels_col[id][i]} is {clf.best_params_}.\n")
            models.append(clf)
            best_params['n_estimators'].append(clf.best_params_['n_estimators'])
        return models, best_params
    
    def __tFixedTrain(self, id, num_estimators=[150]*10):
        assert len(num_estimators) == len(self.st_labels_col[id])
        for i, label in enumerate(self.st1_labels_col):
            clf = RandomForestClassifier(n_estimators=num_estimators[i])
            clf.fit(self.x_train, self.y_st[id][:, i])
            print(f"Score: {metrics.roc_auc_score(self.y_st[id][:, i], clf.predict_proba(self.x_train)[:, 1])}.\n")
            self.st1_mdl.append(clf)
            
    
    def __tCV(self, id, num_estimators=[150]*10):
        assert len(num_estimators) == len(self.st_labels_col[id])
        for i, label in enumerate(self.st_labels_col[id]):
            clf = RandomForestClassifier(n_estimators=num_estimators[i])
            scores = cross_val_score(clf, self.x_train, self.y_st[id][:, i], cv=5, scoring='roc_auc')
            print("Label " + label + " Cross-validation score is {score:.3f},"
                  " standard deviation is {err:.3f}."
                  .format(score=scores.mean(), err=scores.std()))
    
    def __tPredict(self, id, num_estimators):
        for i, label in enumerate(self.st_labels_col[id]):
            clf = self.st_mdl[id][i]
            predictions = clf.predict_proba(self.x_test)[:, 1]
            self.df[label] = predictions
        

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
    task2.task(2, mode='cross_validation', num_estimators=[150])
