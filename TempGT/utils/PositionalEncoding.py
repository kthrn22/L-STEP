import torch
import numpy as np
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.linalg import eigs, eigsh

def LaplacianPE(edge_index, num_nodes, k):
    src, dst = edge_index[0], edge_index[1]

    edge_index, edge_weight = get_laplacian(edge_index = edge_index, num_nodes = num_nodes, normalization = "sym")
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

    eig_fn = eigsh

    eig_vals, eig_vecs = eig_fn(
        L, k = k + 1,
        which = 'SA',
        return_eigenvectors = True
    )

    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = torch.from_numpy(eig_vecs[:, 1: k + 1])
    sign = -1 + 2 * torch.randint(0, 2, (k, ))

    pe *= sign

    return pe

if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1, 1, 2],
                            [1, 0, 2, 1]])
    edge_weight = torch.tensor([1., 2., 2., 4.])

    edge_index, _ = get_laplacian(edge_index)
    print(edge_index)