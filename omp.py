import torch


def omp(A: torch.Tensor, b: torch.Tensor, k, device=None) -> torch.Tensor:
    """
    b = Ax

    :param A: (m, n)
    :param b: (m)
    :param k: sparsity
    :param device: pytorch device
    :return: (n)
    """
    A_m, A_n = A.shape
    r = b
    S = []
    x = torch.zeros((A_n,), device=device)
    for _ in range(k):
        cor = A.T @ r
        ind = torch.argmax(torch.abs(cor))
        S.append(ind)
        index = torch.tensor(S, device=device)
        A_S = torch.index_select(A, 1, index)
        P = A_S @ torch.linalg.pinv(A_S.T @ A_S) @ A_S.T
        r = (torch.eye(A_m, device=device) - P) @ b
    index = torch.tensor(S, device=device)
    A_S = torch.index_select(A, 1, index)
    x_S = torch.linalg.pinv(A_S.T @ A_S) @ A_S.T @ b
    x[index] = x_S
    return x.reshape(A_n)
