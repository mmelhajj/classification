import numpy as np
import scipy.spatial


def compute_cost_matrix(X, Y, metric='euclidean'):
    """Compute the cost matrix of two feature sequences

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        X (np.ndarray): Sequence 1
        Y (np.ndarray): Sequence 2
        metric (str): Cost metric, a valid strings for scipy.spatial.distance.cdist (Default value = 'euclidean')

    Returns:
        C (np.ndarray): Cost matrix
    """
    X, Y = np.atleast_2d(X, Y)
    C = scipy.spatial.distance.cdist(X.T, Y.T, metric=metric)
    return C


def compute_accumulated_cost_matrix(C):
    """Compute the accumulated cost matrix given the cost matrix

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        C (np.ndarray): Cost matrix

    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    N = C.shape[0]
    M = C.shape[1]
    D = np.zeros((N, M))
    D[0, 0] = C[0, 0]
    for n in range(1, N):
        D[n, 0] = D[n - 1, 0] + C[n, 0]
    for m in range(1, M):
        D[0, m] = D[0, m - 1] + C[0, m]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n - 1, m], D[n, m - 1], D[n - 1, m - 1])
    return D


def compute_optimal_warping_path(D):
    """Compute the warping path given an accumulated cost matrix

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        D (np.ndarray): Accumulated cost matrix

    Returns:
        P (np.ndarray): Optimal warping path
    """
    N = D.shape[0]
    M = D.shape[1]
    n = N - 1
    m = M - 1
    P = [(n, m)]
    while n > 0 or m > 0:
        if n == 0:
            cell = (0, m - 1)
        elif m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n - 1, m - 1], D[n - 1, m], D[n, m - 1])
            if val == D[n - 1, m - 1]:
                cell = (n - 1, m - 1)
            elif val == D[n - 1, m]:
                cell = (n - 1, m)
            else:
                cell = (n, m - 1)
        P.append(cell)
        (n, m) = cell
    P.reverse()
    return np.array(P)


def dtw_calculation(df_query, query, template):
    """
    Args:
        df_query (DataFrame): df containing query profile for each plot
        query (str): col name of the query variable
        template (array): col name of the template variable
    Returns:
        cost (float): the coast value
    """
    C = compute_cost_matrix(df_query[query].values, template, metric='euclidean')
    D = compute_accumulated_cost_matrix(C)
    P = compute_optimal_warping_path(D)
    c_P = sum(C[n, m] for (n, m) in P)

    return c_P
