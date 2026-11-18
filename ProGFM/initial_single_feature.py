from sklearn.cluster import KMeans
import torch
from torch_geometric.utils import add_self_loops

def build_edge_relation_repr(x, edge_index, eps=1e-8):
    src, dst = edge_index
    x_src = x[src]
    x_dst = x[dst]
    x_max = x.max(dim=0).values
    x_min = x.min(dim=0).values
    denom = x_max - x_min
    s_edge_dim = 1.0 - torch.abs(x_src - x_dst) / (denom + eps)
    r_edge = s_edge_dim.reshape(-1, 1)
    return r_edge
@torch.no_grad()
def build_relation_prototypes_sklearn_two_stage(
    train_datasets,
    local_num_clusters,
    global_num_clusters,
    device,
    random_state=0,
    max_iter=300,
    eps=1e-8
):
    all_local_centers = []
    for dataset_idx, dataset in enumerate(train_datasets):
        dataset = dataset.to(device)
        edge_index, _ = add_self_loops(
            dataset.edge_index,
            num_nodes=dataset.num_nodes
        )
        r = build_edge_relation_repr(
            x=dataset.x,
            edge_index=edge_index,
            eps=eps
        )
        all_r = r.detach().cpu().numpy()
        n_local_clusters = min(local_num_clusters, all_r.shape[0])
        local_kmeans = KMeans(
            n_clusters=n_local_clusters,
            random_state=random_state,
            n_init="auto",
            max_iter=max_iter
        )
        local_kmeans.fit(all_r)
        local_centers = torch.tensor(
            local_kmeans.cluster_centers_,
            dtype=torch.float32
        )
        all_local_centers.append(local_centers)
    all_local_centers = torch.cat(all_local_centers, dim=0).numpy()
    n_global_clusters = min(global_num_clusters, all_local_centers.shape[0])
    global_kmeans = KMeans(
        n_clusters=n_global_clusters,
        random_state=random_state,
        n_init="auto",
        max_iter=max_iter
    )
    global_kmeans.fit(all_local_centers)
    centers = torch.tensor(
        global_kmeans.cluster_centers_,
        dtype=torch.float32,
        device=device
    )
    return centers