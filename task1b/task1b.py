import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    # read data
    train = pd.read_csv('train.csv')
    train_mat = train.to_numpy()
    y = train_mat[:, 1]
    X = train_mat[:, 2:]

    phi = np.concatenate((X, np.square(X), np.exp(X), np.cos(X), np.ones_like(y).reshape(-1, 1)), axis=1)

    model = LinearRegression(fit_intercept=False)
    model.fit(phi, y)
    print(model.coef_)
