import torch
import numpy as np
from tqdm import tqdm

from omp import omp
import dataset


def src_one(y: torch.Tensor, D: torch.Tensor, *,
            k=None, device=None) -> torch.Tensor:
    """
    y = Dx
    :param y: image (h*w)
    :param D: dict (class_sz, train_im_sz, h*w)
    :param k:
    :param device: pytorch device
    :return: predict tensor(int)
    """
    assert y.dim() == 1
    assert D.dim() == 3
    assert y.size(dim=0) == D.size(dim=2)
    class_sz, train_im_sz, n_features = D.shape  # n_features=h*w
    D_x = D.view(class_sz * train_im_sz, n_features)
    D_x = D_x.permute([1, 0])  # D_x(n_features, class_sz*train_im_sz)
    # y(n_features)
    a = omp(D_x, y, k=k, device=device)  # a(class_sz*train_im_sz)
    X_i = D.permute([0, 2, 1])  # X_i(class_sz, h*w, train_im_sz)
    a_i = a.view(class_sz, train_im_sz, 1)  # a(class_sz, train_im_sz, 1)
    y_p = torch.matmul(X_i, a_i).view(class_sz, n_features)
    e_y = torch.mean((y - y_p) ** 2, dim=1)
    return torch.argmin(e_y)


def src(train_ds: torch.Tensor, test_ds: torch.Tensor, *,
        k=None, device=None) -> np.ndarray:
    """
    Sparse representation-based Classification
    :param train_ds: torch tensor with shape (class_sz, train_im_sz, h, w)
    :param test_ds: torch tensor with shape (class_sz, test_im_sz, h, w)
    :param k: class
    :param device: torch device
    :return: np.ndarray
    """
    assert train_ds.dim() == 4
    assert test_ds.dim() == 4
    assert train_ds.size(dim=0) == test_ds.size(dim=0)
    assert train_ds.size(dim=2) == test_ds.size(dim=2)
    assert train_ds.size(dim=3) == test_ds.size(dim=3)
    class_sz, train_im_sz, h, w = train_ds.size()
    _, test_im_sz, _, _ = test_ds.size()
    D = train_ds.view(class_sz, train_im_sz, h * w)
    ret = np.zeros([class_sz, test_im_sz], dtype=np.int32)
    for i in tqdm(range(class_sz * test_im_sz)):
        test_class = i // test_im_sz
        test_im = i % test_im_sz
        y = test_ds[test_class, test_im, :, :].view(h * w)
        pre = src_one(y, D, k=k, device=device)
        ret[test_class, test_im] = pre.item()
    return ret


def src_eval(train_ds: torch.Tensor, test_ds: torch.Tensor, *,
             k=None, reduction=2, device=None) -> np.ndarray:
    """

    :param train_ds: (class_sz, train_im_sz, h, w)
    :param test_ds: (class_sz, test_im_sz, h, w)
    :param k: n_nonzero_coefs
    :param reduction: 0 or 1 or 2
    :param device: pytorch device
    :return: np.ndarray
    """
    mat = src(train_ds, test_ds, k=k, device=device)
    assert mat.ndim == 2
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

    y = train_ds[0, 0, :, :]
    accu = src_one(y.view(40 * 30), train_ds.view(100, 14, 40 * 30), k=60, device=device)
    print(accu)
