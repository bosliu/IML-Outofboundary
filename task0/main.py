from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

from typing import List, Union, Optional, Tuple

train: pd.DataFrame = pd.read_csv('train.csv')
test: pd.DataFrame = pd.read_csv('test.csv')
# %%
train.head()
# %%
train_mat: np.ndarray = train.to_numpy()
train_y: np.ndarray = train_mat[:, 1]
train_x: np.ndarray = train_mat[:, 2:]

range_x: np.ndarray = np.max(train_x, axis=0) - np.min(train_x, axis=0)
train_x_avg: np.ndarray = np.divide(train_x - np.min(train_x, axis=0), range_x)
x_min = np.min(train_x, axis=0)

range_y: float = np.max(train_y) - np.min(train_y)
train_y_avg: np.ndarray = np.divide(train_y - np.min(train_y), range_y)
y_min = np.min(train_y)


# %%
# class Objective:
#     def __call__(self, w):
#         raise NotImplementedError()
#
#     def grad(self, w):
#         raise NotImplementedError()

class ObjFunc:
    def __init__(self, y: Union[np.ndarray, float], phi: np.ndarray):
        self.y = y.astype(np.float64)
        self.phi = phi.astype(np.float64)  # feature input
        self.n = self.y.shape[0]

    def __call__(self, w: np.ndarray) -> Union[float, np.ndarray]:
        return np.linalg.norm(self.y - self.phi @ w) ** 2 / self.n

    def grad(self, w: np.ndarray) -> np.ndarray:
        return 2 * self.phi.T @ (self.phi @ w - self.y) / self.n


class ObjFuncSDG(ObjFunc):
    def __init__(self, y: Union[np.ndarray, float], phi: np.ndarray):
        super().__init__(y, phi)

    def grad(self, w: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return 2 * self.phi[indices, :].T @ (self.phi[indices, :] @ w - self.y[indices]) / indices.shape[0]


# %%
def gradient_descent(obj_func: ObjFunc, w_init: np.ndarray, lr: float, tol: float, max_steps: int) -> Tuple[
        np.ndarray, List, List]:
    w_curr = w_init
    w_hist: List[np.ndarray] = [w_init]
    obj_hist: List[Union[float, np.ndarray]] = [obj_func(w_curr)]
    error = np.zeros_like(w_curr)
    for step in range(max_steps):
        direction: np.ndarray = obj_func.grad(w_curr)
        w_next = lr * 0.99 * error + w_curr - lr * direction
        error = w_next - w_curr
        # print(obj_func(w_curr))
        if np.allclose(w_next, w_curr, tol):
            break
        w_curr = w_next
        w_hist.append(w_curr)
        obj_hist.append(obj_func(w_curr))
    return w_curr, w_hist, obj_hist


# %%
def sgd(obj_func: ObjFuncSDG, w_init: np.ndarray, lr: float, tol: float, n_epoch: int, batch_size: int) -> np.ndarray:
    w_curr = w_init
    rand_idx: np.ndarray = np.array(
        [x for x in range(obj_func.n)], dtype=int)
    np.random.shuffle(rand_idx)
    error = np.zeros_like(w_curr)
    for epoch in range(n_epoch):
        # lr = max(0.9*lr, 1e-16)
        for i in range(0, obj_func.n, batch_size):
            batch_idx: np.ndarray = rand_idx[i: i + batch_size]
            direction: np.ndarray = obj_func.grad(w_curr, batch_idx)
            # direction = direction / np.linalg.norm(direction)
            w_next = w_curr - lr * direction + 0.8 * error
            error = w_next - w_curr
            # print(obj_func(w_curr))
            if np.linalg.norm(w_next-w_curr) <= tol:
                print("Converged")
                print(lr)
                return w_next
            w_curr = w_next
        print(np.linalg.norm(direction))
    print(lr)
    return w_next


def Newton(obj_func: ObjFuncSDG, w_init: np.ndarray, lr: float, tol: float, n_epoch: int, batch_size: int) -> np.ndarray:
    w_curr = w_init
    batch_idx: np.ndarray = np.array(
        [x for x in range(obj_func.n)], dtype=int)
    error = np.zeros_like(w_curr)
    for epoch in range(n_epoch):
        direction: np.ndarray = obj_func.grad(w_curr, batch_idx)
        w_next = w_curr - \
            np.linalg.inv(obj_func.phi.T  @
                          obj_func.phi) @ direction + 0.8 * error
        error = w_next - w_curr
        if np.linalg.norm(w_next-w_curr) <= tol:
            print("Converged")
            return w_next
        w_curr = w_next
        print("direction: ", np.linalg.norm(direction))
    return w_next

# %%
# hyperparams


learning_rate = 0.001
tol = 1e-12
# n_steps = 100

batch_size = 100
num_of_epochs = 19000
# %%
phi = train_x_avg.copy().astype(np.float64)
phi = np.insert(phi, 0, 1, axis=1)
# %%
w_init = np.random.random(11)
fitting = ObjFuncSDG(y=train_y_avg, phi=phi)
w_res = Newton(fitting, w_init, lr=learning_rate, tol=tol,
               n_epoch=num_of_epochs, batch_size=batch_size)

# use scikit_learn to compare the results
reg = LinearRegression().fit(phi, train_y_avg)
range_x = np.insert(range_x, 0, 1)
# parameters recover
w = np.multiply(range_y, reg.coef_) / range_x
print("sklearn result: ", w)

# Newton's method
# parameters recover
w_sgd = range_y * w_res / range_x
w_sgd[0] += - (w_sgd[1:] @ x_min) + y_min
print("Newton's result: ", w_sgd)
