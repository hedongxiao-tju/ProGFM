
from utils.Classifier import Classifier
from utils.seed import set_seed
from utils.get_train_test_data import get_data
import numpy as np

import torch.optim as optim
import sys
import torch
import argparse
import warnings
from model_single_feature import center_away_loss, neighbor_close_loss
from model_single_feature import EdgeTokenGCN
from initial_single_feature import build_relation_prototypes_sklearn_two_stage

def main():
    def train():
        gcn.train()
        optimizer.zero_grad()
        z = gcn(
            h=train_data.x,
            edge_index=train_data.edge_index,
        )
        loss1 = center_away_loss(z)
        loss2 = neighbor_close_loss(z,train_data.edge_index)
        loss_train = (loss1 + args.loss_ratio * loss2)
        loss_train.backward()
        optimizer.step()
        print('Epoch: {:04d}'.format(epoch + 1),'loss_train: {:.4f}'.format(loss_train.item()))
    def test():
        gcn.eval()
        with torch.no_grad():

            z = gcn(
                h=test_data.x,
                edge_index=test_data.edge_index,
            )
        log = Classifier(ft_in=z.shape[1], nb_classes=torch.max(test_data.y).item() + 1, args = args)
        num_trials =500
        shot_num = args.shot_num
        acc_list = []
        for i in range(num_trials):
            sample_data_foler_path = args.datasets_dir + "Experiment/sample_data/Node/{}/{}_shot/{}".format(args.test_dataset, shot_num, i + 1)
            idx_train = torch.load(f"{sample_data_foler_path}/train_idx.pt").type(torch.long).cuda(args.gpu)
            idx_test = torch.load(f"{sample_data_foler_path}/test_idx.pt").type(torch.long).cuda(args.gpu)
            log.forward(z[idx_train], test_data.y[idx_train], train=1).float().cuda(args.gpu)
            batch_size = 100
            total_correct = 0
            total_samples = 0
            for i in range(0, len(idx_test), batch_size):
                batch_indices = idx_test[i:i + batch_size]
                logits = log.forward(z[batch_indices], test_data.y[batch_indices]).float().cuda(args.gpu)
                preds = torch.argmax(logits, dim=1).cuda(args.gpu)
                total_correct += torch.sum(preds == test_data.y[batch_indices]).item()
                total_samples += len(batch_indices)
            acc = total_correct / total_samples
            acc_list.append(acc)
        average_acc = sum(acc_list) / num_trials
        std_acc = np.std(acc_list)
        print(f"Average ACC over {num_trials} trials: {average_acc:.4f} + {std_acc:.4f}")
        print("-------------------------------------------------------------------------------------")
    train_dataset, test_data = get_data(args)
    centers = build_relation_prototypes_sklearn_two_stage(train_dataset, args.num_hypernodes, args.num_hypernodes, args.gpu, random_state=0)
    gcn = EdgeTokenGCN(args.svd_k, args.hidden_dimension,args.out_dimension,centers, tau=1.0, hard=True)
    if torch.cuda.is_available():
        gcn.cuda(args.gpu)
    optimizer = optim.Adam(gcn.parameters(),  lr=args.lr,  weight_decay=args.weight_decay)
    for train_data in train_dataset: train_data.cuda(args.gpu)
    for epoch in range(args.epochs):
        for train_data in train_dataset:
            train()
    for train_data in train_dataset: train_data.detach().cpu()
    optimizer.zero_grad()
    test_data.cuda(args.gpu)
    test()
    test_data.detach().cpu()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='seed.')#86
    parser.add_argument('--gpu', type=int, default=6, help='gpu')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--loss_ratio', type=float, default=0.2, help='Proportion of loss composition')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs to train.')
    parser.add_argument('--shot_num', type=int, default=1, help='shot_num')
    parser.add_argument('--svd_k', type=int, default=16, help='SVD_out_dimension')
    parser.add_argument('--hidden_dimension', type=int, default=256, help='GCN_hidden_dimension')
    parser.add_argument('--out_dimension', type=int, default=128, help='GCN_final_out_dimension')
    parser.add_argument('--num_hypernodes', type=int, default=100, help='num_hypernodes')

    parser.add_argument('--datasets_dir', type=str, default='../../', help='datasets dir.')
    parser.add_argument('--train_datasets', type=str, nargs='+',default=['Cora','CiteSeer','PubMed', 'Computers','CS',], help='train datasets.')
    parser.add_argument('--test_dataset', type=str,default= 'Photo', help='test dataset.')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    set_seed(args.seed)
    main();