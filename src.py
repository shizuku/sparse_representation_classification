import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple

from omp import omp
from utils import distance


def src_one(y: torch.Tensor, A: torch.Tensor, k=1, device=None):
    """

    :param y: (h*w)
    :param A: (class_sz, im, h*w)
    :param k:
    :param device: pytorch device
    :return:
    """
    class_sz, _, _ = A.shape
    mat = torch.zeros((class_sz,), dtype=torch.float32, device=device)
    for cls in range(class_sz):
        A_cls = A[cls, :, :].T
        x = omp(A_cls, y, k, device=device)
        pre_y = A_cls @ x
        d = distance(pre_y, y)
        mat[cls] = d
    return torch.argmin(mat).item()


def src(train_ds: torch.Tensor, test_ds: torch.Tensor,
        k=1, device=None) -> Tuple[float, np.ndarray]:
    """

    :param train_ds: (class_sz, train_im_sz, h, w)
    :param test_ds: (class_sz, test_im_sz, h, w)
    :param k:
    :param device: pytorch device
    :return: Tuple[float, np.ndarray]
    """
    class_sz, train_im_sz, h, w = train_ds.shape
    _, test_im_sz, _, _ = test_ds.shape
    A = train_ds.reshape((class_sz, train_im_sz, h * w))
    ret = np.zeros((class_sz,), dtype=np.float32)
    for test_cls in tqdm(range(class_sz)):
        correct = 0
        total = 0
        for test_im in range(test_im_sz):
            y = test_ds[test_cls, test_im, :, :].reshape(h * w)
            pre = src_one(y, A, k=k, device=device)
            if pre == test_cls:
                correct += 1
            total += 1
        ret[test_cls] = correct / total
    return ret.mean(), ret
