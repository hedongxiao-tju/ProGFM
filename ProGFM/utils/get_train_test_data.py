from FLT.Dataset_Load import load_dataset
import numpy as np
import scipy as sp
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_undirected
def get_data(args):
    with torch.no_grad():
        train_dataset = []
        for dataset_name in args.train_datasets:
            train_data = load_dataset(dataset_name, args.datasets_dir, args)[0]
            train_data.edge_index = to_undirected(train_data.edge_index)
            train_data.x = svd(train_data.x, args)
            train_data = train_data.detach().cpu()
            train_dataset.append(train_data)
        test_data = load_dataset(args.test_dataset, args.datasets_dir, args)[0]
        test_data.edge_index = to_undirected(test_data.edge_index)
        test_data.x = svd(test_data.x, args)
        test_data = test_data.detach().cpu()
    return train_dataset, test_data

def svd(x, args):
    x = x.numpy()

    n_features = x.shape[1]
    n_samples = x.shape[0]
    n_samples_beifen = x.shape[0]
    if n_features < args.svd_k:
        repeat_times = int(np.floor(args.svd_k / n_features)) + 1
        x = np.tile(x, (1, repeat_times))
    if n_samples < args.svd_k:
        repeat_times = int(np.ceil(args.svd_k / n_samples))
        x = np.tile(x, (repeat_times, 1))
    row_min = x.min(axis=1)
    row_max = x.max(axis=1)
    row_range = row_max - row_min
    row_range[row_range == 0] = 1
    x = (x - row_min[:, np.newaxis]) / row_range[:, np.newaxis]
    U, S, Vt = sp.linalg.svds(x, k=args.svd_k, random_state=args.seed)
    U = torch.tensor(U.copy(), dtype=torch.float32).cuda(args.gpu)
    S = torch.tensor(S.copy(), dtype=torch.float32).cuda(args.gpu)
    sqrt_Lambda_S = torch.sqrt(torch.diag(S))
    result = U @ sqrt_Lambda_S
    return result[:n_samples_beifen].detach().cpu()
