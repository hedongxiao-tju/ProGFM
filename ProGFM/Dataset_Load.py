import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Reddit, LastFMAsia, FacebookPagePage
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork, ppi
from torch_geometric.datasets import HeterophilousGraphDataset
import os.path as osp
def load_dataset(dataset_name, dataset_dir, args):
    dataset_dir = dataset_dir + 'datasets'
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name=dataset_name,
                            transform=T.NormalizeFeatures())
    elif dataset_name in ['Photo', 'Computers']:
        dataset = Amazon(dataset_dir, name=dataset_name,
                         transform=T.NormalizeFeatures())

    elif dataset_name in ['CS',]:
        dataset = Coauthor(dataset_dir, name=dataset_name,
                           transform=T.NormalizeFeatures())
    return dataset




