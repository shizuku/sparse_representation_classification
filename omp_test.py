import numpy as np
import torch
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.datasets import make_sparse_coded_signal
from omp import omp as my_omp
import unittest

n_components, n_features = 512, 100
n_nonzero_coefs = 17


def sk_omp(X: np.ndarray, y: np.ndarray, *, k=None) -> np.ndarray:
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k,
                                    normalize=False,
                                    fit_intercept=False,
                                    precompute=False)
    omp.fit(X, y)
    return omp.coef_


def np_eq(a: np.ndarray, b: np.ndarray, eps=1e-5) -> bool:
    return (((a - b) ** 2) < eps).all()


class TestOmp(unittest.TestCase):
    def test_omp(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for _ in range(1000):
            y, X, w = make_sparse_coded_signal(
                n_samples=1,
                n_components=n_components,
                n_features=n_features,
                n_nonzero_coefs=n_nonzero_coefs,
            )
            # X(n_features, n_components)
            # y(n_features)
            # w(n_components)
            sk_x = sk_omp(X, y, k=n_nonzero_coefs)
            my_x = my_omp(torch.tensor(X, dtype=torch.float32, device=device),
                          torch.tensor(y, dtype=torch.float32, device=device),
                          k=n_nonzero_coefs, device=device).detach().numpy()
            self.assertTrue(np_eq(sk_x, my_x, 1e-10))
