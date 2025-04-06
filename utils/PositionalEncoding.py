import torch
import numpy as np
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.linalg import eigs, eigsh
from torch_geometric.utils import scatter as tg_scatter
from torch_geometric.utils import is_torch_sparse_tensor, to_torch_csr_tensor, get_self_loop_attr, to_edge_index

def LaplacianPE_2(src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_ids: np.ndarray, num_nodes: int, k: int):
    mapping_dict = {}
    for idx, val in enumerate(node_ids):
        mapping_dict[val] = idx
    
    mapped_src = np.array([mapping_dict[i] for i in src_node_ids])
    mapped_dst = np.array([mapping_dict[i] for i in dst_node_ids])
    
    edge_index = torch.from_numpy(np.array([mapped_src.tolist() + mapped_dst.tolist(),
                                            mapped_dst.tolist() + mapped_src.tolist()]))
    
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

    return pe, edge_weight

def get_pe(out, N):
    # if is_torch_sparse_tensor(out):
    return get_self_loop_attr(*to_edge_index(out), num_nodes=N)
    # return out[loop_index, loop_index]

def RandomWalkPE(edge_index, num_nodes, walk_length):
    row, col = edge_index
    N = num_nodes

    num_edges = edge_index.shape[-1]

    value = torch.ones(num_edges, device=row.device)
    
    value = tg_scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
    value = 1.0 / value

    
    adj = to_torch_csr_tensor(edge_index, value, size=N)

    out = adj
    pe_list = [get_pe(out, N = N)]
    for _ in range(walk_length - 1):
        out = out @ adj
        pe_list.append(get_pe(out, N = N))

    pe = torch.stack(pe_list, dim=-1)

    return pe

if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1, 1, 2],
                            [1, 0, 2, 1]])
    edge_weight = torch.tensor([1., 2., 2., 4.])

    edge_index, _ = get_laplacian(edge_index)
    print(edge_index)