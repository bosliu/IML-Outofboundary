import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from typing import Optional

# hyper-parameters
Lmbda = [10000, 1, 10, 100, 200]
K = 10  # num of fold
LR = 1e-5
TOL = 1e-6
BATCH_SIZE = 15


class Regression:
    def __init__(self, phi: np.ndarray, y: np.ndarray):
        self.phi = phi
        self.y = y
        self.n = self.y.shape[0]
        self.w = None

    def grad(self, w: np.ndarray, indices: np.ndarray, lmbda: float = 0, ridge: bool = True):
        # by default: set ridge option to True
        return 2 * self.phi[indices, :].T @ (self.phi[indices, :] @ w - self.y[indices]) / indices.shape[0] \
               + ridge * 2 * lmbda * w

    def loss(self):
        pass

    def fit_sgd(self, data_range: np.ndarray, w0: np.ndarray, lr: float, tol: float, n_epoch: int, batch_size: int,
                lmbda: float = 0, return_value: bool = False, momentum: bool = True) -> Optional[np.ndarray]:
        w_curr = w0
        self.w = np.zeros_like(w0)  # reset
        rand_idx: np.ndarray = data_range
        error = np.zeros_like(w_curr)
        for epoch in range(n_epoch):
            np.random.shuffle(rand_idx)
            for i in range(0, data_range.shape[0], batch_size):
                batch_idx: np.ndarray = rand_idx[i: i + batch_size]
                direction: np.ndarray = self.grad(w_curr, batch_idx, lmbda=lmbda)
                w_next = w_curr - lr * direction + 0.8 * error * momentum
                error = w_next - w_curr
                if np.linalg.norm(error) <= tol:
                    print("Converged")
                    self.w = w_next
                    if return_value:
                        return self.w
                w_curr = w_next
        self.w = w_next
        if return_value:
            return self.w

    def Newton(self, w0: np.ndarray, lr: float, tol: float, n_epoch: int,
               lmbda: float = 0, return_value: bool = False,) -> Optional[np.ndarray]:
        w_curr = w0
        self.w = np.zeros_like(w0)  # reset
        batch_idx: np.ndarray = np.array(
            [x for x in range(self.n)], dtype=int)
        error = np.zeros_like(w_curr)
        for epoch in range(n_epoch):
            direction: np.ndarray = self.grad(w_curr, batch_idx, lmbda=lmbda)
            w_next = w_curr - \
                     np.linalg.inv(self.phi.T @
                                   self.phi) @ direction + 0.8 * error
            error = w_next - w_curr
            if np.linalg.norm(w_next - w_curr) <= tol:
                print("Converged")
                self.w = w_next
                if return_value:
                    return self.w
            w_curr = w_next
            # print("direction: ", np.linalg.norm(direction))
        self.w = w_next
        if return_value:
            return self.w

    def eval(self, indices):
        return mean_squared_error(self.y[indices], self.phi[indices] @ self.w)

    def reset_weight(self):
        self.w = None


if __name__ == '__main__':
    # read data
    train: pd.DataFrame = pd.read_csv('train.csv')
    train_mat: np.ndarray = train.to_numpy()
    train_y: np.ndarray = train_mat[:, 0]
    train_x: np.ndarray = train_mat[:, 1:]

    # asked to perform linear regression on original features
    phi = np.insert(train_x.copy().astype(np.float64), 0, 1, axis=1)

    # initialization
    w_init = np.random.random(phi.shape[1])
    rmses = []
    fold_indices = np.array_split(np.arange(train_y.shape[0]), K)

    model = Regression(phi, train_y)

    for l in Lmbda:
        rmse: float = 0
        print(f"{l}\n")
        for folded in fold_indices:
            model.reset_weight()
            data = np.asarray(list(set(np.arange(train_y.shape[0])) - set(folded)))
            model.Newton(w_init, LR, TOL, 10000, l)
            # model.fit_sgd(data, w_init, lr=LR, tol=TOL, n_epoch=10000, batch_size=BATCH_SIZE, lmbda=l, momentum=True)
            rmse += model.eval(folded)  # todo fold k!
        rmses.append(rmse/K)

    print(rmses)