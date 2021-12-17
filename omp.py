import torch


def omp(X: torch.Tensor, y: torch.Tensor, *,
        k=None, device=None) -> torch.Tensor:
    """
    y = Xa

    :param X: (n_features, n_components)
    :param y: (n_features)
    :param k: n_nonzero_coefs
    :param device: pytorch device
    :return: (n_components)
    """
    assert X.dim() == 2
    assert y.dim() == 1
    assert X.size(dim=0) == y.size(dim=0)
    n_features, n_components = X.size()
    if k is None:
        k = n_features // 10
    r = y
    S = []
    a = torch.zeros((n_components,), device=device)
    for _ in range(k):
        cor = X.T @ r
        ind = torch.argmax(torch.abs(cor))
        S.append(ind)
        index = torch.tensor(S, device=device)
        A_S = torch.index_select(X, 1, index)
        P = A_S @ torch.linalg.pinv(A_S.T @ A_S) @ A_S.T
        r = (torch.eye(n_features, device=device) - P) @ y
    index = torch.tensor(S, device=device)
    A_S = torch.index_select(X, 1, index)
    x_S = torch.linalg.pinv(A_S.T @ A_S) @ A_S.T @ y
    a[index] = x_S
    return a.reshape(n_components)
