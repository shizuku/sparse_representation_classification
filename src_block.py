import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple

from omp import omp
from utils import distance, vote


def src_block_one(y: torch.Tensor, A: torch.Tensor, k=1, device=None):
    """

    :param y: (block_sz, h*w)
    :param A: (block_sz, class_sz, im, h*w)
    :param k:
    :param device: pytorch device
    :return:
    """
    block_sz, _ = y.shape
    _, class_sz, _, _ = A.shape
    mat = torch.zeros((block_sz, class_sz,), dtype=torch.float32, device=device)
    for blk in range(block_sz):
        blk_mat = []
        for cls in range(class_sz):
            blk_y = y[blk, :]
            blk_A = A[blk, cls, :, :].T
            blk_x = omp(blk_A, blk_y, k, device=device)
            blk_pre_y = blk_A @ blk_x
            d = distance(blk_pre_y, blk_y)
            blk_mat.append(d)
            mat[blk, cls] = d
    return vote(torch.argmin(mat, dim=1), clazz=class_sz).item()


def src_block(train_ds: torch.Tensor, test_ds: torch.Tensor,
              k=1, device=None) -> Tuple[float, np.ndarray]:
    """

    :param train_ds: (block_sz, class_sz, train_im_sz, h, w)
    :param test_ds: (block_sz, class_sz, test_im_sz, h, w)
    :param k:
    :param device: pytorch device
    :return: Tuple[float, np.ndarray]
    """
    block_sz, class_sz, train_im_sz, h, w = train_ds.shape
    _, _, test_im_sz, _, _ = test_ds.shape
    A = train_ds.reshape((block_sz, class_sz, train_im_sz, h * w))
    ret = np.zeros((class_sz,), dtype=np.float32)
    for test_cls in tqdm(range(class_sz)):
        correct = 0
        total = 0
        for test_im in range(test_im_sz):
            y = test_ds[:, test_cls, test_im, :, :].reshape(block_sz, h * w)
            pre = src_block_one(y, A, k=k, device=device)
            if pre == test_cls:
                correct += 1
            total += 1
        ret[test_cls] = correct / total
    return ret.mean(), ret
