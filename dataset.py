import numpy as np
from utils import read_pgm, average_pool
from sklearn.preprocessing import normalize

FS = "../ar/{wm}-{i:03}-{j:02}.pgm"

SP = (40, 30)


def down_sample(img: np.ndarray) -> np.ndarray:
    return average_pool(img, (4, 4), (4, 4))[1:, :]


def norm(img: np.ndarray, eps=1e-5) -> np.ndarray:
    # return (img - img.mean()) / np.sqrt(img.var() + eps)
    # return (img / 255) - 0.5
    return normalize(img)


def read_train() -> np.ndarray:
    A = []
    for wm in ['m', 'w']:
        for i in range(1, 51):
            A_i = []
            for j in range(1, 8):
                a = read_pgm(FS.format(wm=wm, i=i, j=j))
                a = norm(down_sample(a)).reshape(SP)
                A_i.append(a)
            for j in range(14, 21):
                a = read_pgm(FS.format(wm=wm, i=i, j=j))
                a = norm(down_sample(a)).reshape(SP)
                A_i.append(a)
            A.append(A_i)
    return np.array(A).astype(np.float32)


def read_test() -> np.ndarray:
    A = []
    for wm in ['m', 'w']:
        for i in range(1, 51):
            A_i = []
            for j in range(8, 14):
                a = read_pgm(FS.format(wm=wm, i=i, j=j))
                a = norm(down_sample(a)).reshape(SP)
                A_i.append(a)
            for j in range(21, 27):
                a = read_pgm(FS.format(wm=wm, i=i, j=j))
                a = norm(down_sample(a)).reshape(SP)
                A_i.append(a)
            A.append(A_i)
    return np.array(A).astype(np.float32)


def train() -> np.ndarray:
    return np.load("ar_train.npy")


def test() -> np.ndarray:
    return np.load("ar_test.npy")


if __name__ == '__main__':
    train_ds = read_train()
    np.save("ar_train.npy", train_ds)

    test_ds = read_test()
    np.save("ar_test.npy", test_ds)
