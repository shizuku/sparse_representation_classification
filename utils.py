import numpy as np
import math


def read_pgm(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        f.readline()  # P5\n
        (width, height) = [int(i) for i in f.readline().split()]
        depth = int(f.readline())
        assert depth <= 255
        mat = []
        for _ in range(height):
            row = []
            for _ in range(width):
                row.append(ord(f.read(1)))
            mat.append(row)
        return np.array(mat)


def im2col(x: np.ndarray, kernel_size, stride) -> np.ndarray:
    h_i, w_i = x.shape
    h_k, w_k = kernel_size
    ret = []
    for i in range(0, h_i - h_k + 1, stride[0]):
        for j in range(0, w_i - w_k + 1, stride[1]):
            m = x[i:i + h_k, j:j + w_k]
            ret.append(m.reshape(1, h_k * w_k))
    return np.concatenate(ret, axis=0)


def max_pool(x: np.ndarray, kernel_size, stride) -> np.ndarray:
    h_i, w_i = x.shape
    h_k, w_k = kernel_size
    h_o = math.floor((h_i - (h_k - 1) - 1) / stride[0] + 1)
    w_o = math.floor((w_i - (w_k - 1) - 1) / stride[1] + 1)
    col = im2col(x, kernel_size, stride)
    return np.max(col, axis=-1).reshape(h_o, w_o)


def average_pool(x: np.ndarray, kernel_size, stride) -> np.ndarray:
    h_i, w_i = x.shape
    h_k, w_k = kernel_size
    h_o = math.floor((h_i - (h_k - 1) - 1) / stride[0] + 1)
    w_o = math.floor((w_i - (w_k - 1) - 1) / stride[1] + 1)
    col = im2col(x, kernel_size, stride)
    return np.mean(col, axis=-1).reshape(h_o, w_o)
