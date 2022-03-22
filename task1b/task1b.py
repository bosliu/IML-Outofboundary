import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    # read data
    train = pd.read_csv('train.csv')
    train_mat = train.to_numpy()
    y = train_mat[:, 1]
    X = train_mat[:, 2:]

    x1, x2, x3, x4, x5 = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4], X[:, 4:5]
    xs = (x1, x2, x3, x4, x5)

    quad = lambda xseq: tuple(np.square(x) for x in xseq)
    expo = lambda xseq: tuple(np.exp(x) for x in xseq)
    cosine = lambda xseq: tuple(np.cos(x) for x in xseq)

    phi1 = np.concatenate(xs, axis=1)
    phi2 = np.concatenate(quad(xs), axis=1)
    phi3 = np.concatenate(expo(xs), axis=1)
    phi4 = np.concatenate(cosine(xs), axis=1)
    phi5 = np.ones_like(y).reshape(-1, 1)
    phi = np.concatenate((phi1, phi2, phi3, phi4, phi5), axis=1)

    model = LinearRegression()
    model.fit(phi, y)
    print(model.coef_)
