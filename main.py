import torch
import argparse
from matplotlib import pyplot as plt

import dataset
from src import src
from src_block import src_block


def split4(x: torch.Tensor):
    row_p = [0, 15, 30]
    col_p = [0, 10, 20, 30, 40]
    res = []
    for j in range(len(col_p) - 1):
        for i in range(len(row_p) - 1):
            m = x[:, :, col_p[j]:col_p[j + 1], row_p[i]:row_p[i + 1]]
            res.append(m)
    return torch.stack(res)


def plot_save(hist, filename):
    plt.bar(range(len(hist)), hist)
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparse Representation Classification')
    parser.add_argument('--no-block', action='store_true', default=False,
                        help='disables block')
    parser.add_argument('--use-train', action='store_true', default=False,
                        help='test use train dataset')
    parser.add_argument('--k', type=int, default=100, help='sparsity')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    train_ds = dataset.train()
    train_ds = torch.tensor(train_ds, device=device)

    test_ds = dataset.test()
    test_ds = torch.tensor(test_ds, device=device)

    if args.no_block:
        fn = src
    else:
        fn = src_block
        train_ds = split4(train_ds)
        test_ds = split4(test_ds)

    if args.use_train:
        test_ds = train_ds

    accu, hist = fn(train_ds, test_ds, k=args.k, device=device)
    print(accu)
    plot_save(hist, f'ar{"" if args.no_block else "_block"}_{args.k}.png')
