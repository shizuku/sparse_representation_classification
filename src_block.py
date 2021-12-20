import torch
import numpy as np
from typing import List

from src import src
import dataset


def split4(x: torch.Tensor):
    row_p = [0, 15, 30]
    col_p = [0, 10, 26, 32, 40]
    res = []
    for j in range(len(col_p) - 1):
        for i in range(len(row_p) - 1):
            m = x[:, :, col_p[j]:col_p[j + 1], row_p[i]:row_p[i + 1]]
            res.append(m.contiguous())
    return res


def vote1(x: np.ndarray, class_sz) -> np.ndarray:
    assert x.ndim == 1
    block_sz = x.shape[0]
    arr = np.zeros([class_sz])
    for b in range(block_sz):
        arr[x[b]] += 1
    return arr.argmax()


def vote(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 3  # (block_sz, class_sz, im_sz)
    block_sz, class_sz, im_sz = arr.shape
    ret = np.zeros([class_sz, im_sz])
    for clazz in range(class_sz):
        for im in range(im_sz):
            b = arr[:, clazz, im]
            ret[clazz, im] = vote1(b, class_sz=class_sz)
    return np.array(ret)


def src_block_one():
    pass


def src_block(train_ds: List[torch.Tensor], test_ds: List[torch.Tensor], *,
              k=None, device=None) -> np.ndarray:
    """

    :param train_ds: [(class_sz, train_im_sz, h, w)]
    :param test_ds: [(class_sz, test_im_sz, h, w)]
    :param k: n_nonzero_coefs
    :param device: pytorch device
    :return: Tuple[float, np.ndarray]
    """
    assert len(train_ds) == len(test_ds)
    block_arr = []
    for train_b, test_b in zip(train_ds, test_ds):
        mat = src(train_b, test_b, k=k, device=device)
        assert mat.ndim == 2
        block_arr.append(mat)
    block_arr = np.array(block_arr)
    arr = vote(block_arr)
    return arr


def src_block_eval(train_ds: List[torch.Tensor], test_ds: List[torch.Tensor], *,
                   k=None, reduction=2, device=None) -> np.ndarray:
    """

    :param train_ds: [(class_sz, train_im_sz, h, w)]
    :param test_ds: [(class_sz, test_im_sz, h, w)]
    :param k: n_nonzero_coefs
    :param reduction: 0 or 1 or 2
    :param device: pytorch device
    :return: Tuple[float, np.ndarray]
    """
    mat = src_block(train_ds, test_ds, k=k, device=device)
    class_sz, _ = mat.shape
    label = np.arange(0, class_sz).reshape(class_sz, 1)  # (class_sz, 1)
    arr = np.where(mat == label, 1, 0)

    if reduction == 2:
        return arr
    if reduction == 1:
        return arr.mean(axis=1)
    if reduction == 0:
        return arr.mean()
    else:
        raise ValueError("reduction must be 0 or 1 or 2")


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    train_ds = dataset.train()
    train_ds = torch.tensor(train_ds, device=device)

    test_ds = dataset.test()
    test_ds = torch.tensor(test_ds, device=device)

    train_ds = split4(train_ds)
    test_ds = split4(test_ds)
    accu = src_block_eval(train_ds, test_ds, k=60, device=device)
    print(accu)
