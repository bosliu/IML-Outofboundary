import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

# hyper-parameters
Lmbda = [0.1, 1, 10, 100, 200]
K = 10  # num of fold

if __name__ == '__main__':
    # read data
    train = pd.read_csv('train.csv')
    train_mat = train.to_numpy()
    train_y = train_mat[:, 0]
    train_x = train_mat[:, 1:]

    # asked to perform linear regression on original features
    phi = np.insert(train_x.copy().astype(np.float64), 0, 1, axis=1)

    # initialization
    rmses = []
    fold_indices = np.array_split(np.arange(train_y.shape[0]), K)

    for l in Lmbda:
        model = Ridge(alpha=l, fit_intercept=False)
        rmse: float = 0
        for folded in fold_indices:
            data = np.asarray(list(set(np.arange(train_y.shape[0])) - set(folded)))
            model.fit(phi[data], train_y[data])
            rmse += mean_squared_error(train_y[folded], model.predict(phi[folded]), squared=False)
        rmses.append(rmse/K)

    print(rmses)
    df = pd.DataFrame(rmses)
    df.to_csv('result.csv', index=False, header=False)
