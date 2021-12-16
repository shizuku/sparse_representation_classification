import numpy as np
import torch
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.datasets import make_sparse_coded_signal
from omp import omp as my_omp

import matplotlib.pyplot as plt

n_components, n_features = 512, 100
n_nonzero_coefs = 17

y, X, w = make_sparse_coded_signal(
    n_samples=1,
    n_components=n_components,
    n_features=n_features,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=0,
)
y_noisy = y + 0.05 * np.random.randn(len(y))


def sk_omp(A: np.ndarray, b: np.ndarray, k):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, normalize=False)
    omp.fit(A, b)
    return omp.coef_


def plot_x(x, i=0, t=5):
    (idx_r,) = x.nonzero()
    plt.subplot(t, 1, i)
    plt.xlim(0, 512)
    plt.stem(idx_r, x[idx_r], use_line_collection=True)


if __name__ == '__main__':
    sk_x = sk_omp(X, y, n_nonzero_coefs)
    my_x = my_omp(torch.tensor(X, dtype=torch.float32),
                  torch.tensor(y, dtype=torch.float32), n_nonzero_coefs)
    sk_x_n = sk_omp(X, y_noisy, n_nonzero_coefs)
    my_x_n = my_omp(torch.tensor(X, dtype=torch.float32),
                    torch.tensor(y_noisy, dtype=torch.float32), n_nonzero_coefs)

    plot_x(w, 1, 5)
    plot_x(sk_x, 2, 5)
    plot_x(my_x.detach().numpy(), 3, 5)
    plot_x(sk_x_n, 4, 5)
    plot_x(my_x_n.detach().numpy(), 5, 5)
    plt.subplots_adjust(0.06, 0.04, 0.94, 0.90, 0.20, 0.38)
    plt.show()
