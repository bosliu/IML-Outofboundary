import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

from typing import Optional

# hyper-parameters
Lmbda = [0.1, 1, 10, 100, 200]
K = 10  # num of fold
LR = 1e-5
TOL = 1e-6
BATCH_SIZE = 15

if __name__ == '__main__':
    # read data
    train: pd.DataFrame = pd.read_csv('train.csv')
    train_mat: np.ndarray = train.to_numpy()
    train_y: np.ndarray = train_mat[:, 0]
    train_x: np.ndarray = train_mat[:, 1:]

    # asked to perform linear regression on original features
    phi = np.insert(train_x.copy().astype(np.float64), 0, 1, axis=1)

    # initialization
    #w_init = np.random.random(phi.shape[1])
    rmses = []
    fold_indices = np.array_split(np.arange(train_y.shape[0]), K)

    # model = Regression(phi, train_y)

    for l in Lmbda:
        model = Ridge(alpha=l, fit_intercept=False)
        rmse: float = 0
        print(f"{l}\n")
        for folded in fold_indices:
            data = np.asarray(list(set(np.arange(train_y.shape[0])) - set(folded)))
            model.fit(phi[data], train_y[data])
            print(model.coef_)
            rmse += mean_squared_error(train_y[folded], model.predict(phi[folded]), squared=False)  # todo fold k!
        rmses.append(rmse/K)

    print(rmses)
    df = pd.DataFrame(rmses)
    df.to_csv(r'C:/course/IML/IML-Outofboundary/task1a/result.csv', index = False)