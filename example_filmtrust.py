import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import Clustering
from Rec_User import *
from Evaluate import *
import matplotlib.pyplot as plt

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim) #设置网络中的全连接层
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5) #对某一维度进行归一化
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        #return rating
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)  #rating
        return self.criterion(scores, labels_list)



def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    outputfile = open('result_cluster.txt', mode='a', encoding='utf-8')
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae),file=outputfile)
            running_loss = 0.0
    outputfile.close()
    return 0

################################################修改中#################################
def test(model, device, test_loader,k,margin_users):
    model.eval()
    tmp_pred = []
    target = []
    all_info=defaultdict(list)
    marginal_info=defaultdict(list)
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)  #test_u to test_v's rating
            for i in range(len(test_u)):
                all_info[test_u.numpy().tolist()[i]].append((test_v.numpy().tolist()[i],val_output.numpy().tolist()[i],tmp_target.numpy().tolist()[i]))
                #for marginal users
                if test_u.numpy().tolist()[i] in margin_users:
                    marginal_info[test_u.numpy().tolist()[i]].append((test_v.numpy().tolist()[i],val_output.numpy().tolist()[i],tmp_target.numpy().tolist()[i]))
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
        margin_ndgc, margin_hr = Evaluate().dealt(k, marginal_info)
        ndgc,hr=Evaluate().dealt(k, all_info)
    tmp_pred = np.array(sum(tmp_pred, []))   #预测的rating
    target = np.array(sum(target, []))   #真实的rating
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae,ndgc, hr, margin_ndgc, margin_hr


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--max_k',type=int,default=4,metavar='N',help='k-means')
    args = parser.parse_args()

    '''os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")'''
    device="cpu"

    embed_dim = args.embed_dim
    dir_data = './data/example_filmtrust_marginal'

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, ratings_list, social_adj_lists,super_users,bridge_users,common_users, margin_users= pickle.load(
        data_file)
    """
    ## toy dataset 
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)
    
    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)
    
    # please add the validation set
    
    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
    
    super_users,bridge_users,common_users
    """

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=6)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True,num_workers=6)
    num_users = max(history_u_lists.__len__(),1643)  #1508: the max id of users in UV_graph; 1642: the max id of users in social_graph
    num_items = max(history_v_lists.__len__(),2072)  #2071: the max id of items
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    #print(u2e.__sizeof__())
    v2e= nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)

    #social info
    #whether in one of the clusters
    common_user_list = defaultdict(int)
    for i in common_users:
        common_user_list[i] = 0
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    cluster_list = Clustering.Clustering(super_users,social_adj_lists,args.max_k,common_user_list).ckmeans()
    enc_u = Social_Encoder(super_users,cluster_list,lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)
    #Rec_User(enc_u, super_users, cluster_list, common_users, common_user_list)
    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)
    # model
    graphrec = GraphRec(enc_u_history, enc_v_history, r2e).to(device)
    #graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0
    outputfile = open('result_cluster.txt', mode='a', encoding='utf-8')
    ndgc_k=10   #ndgc中参数k的设置：10，20，30 也是hr_k
    for epoch in range(1, args.epochs + 1):
        # update cluster_list
        '''if epoch!=1:
            Rec_User(enc_u,super_users,cluster_list,common_users,common_user_list)'''
        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        '''expected_rmse, mae, ndgc,hr= test(graphrec, device, test_loader,ndgc_k,)'''
        expected_rmse, mae, ndgc, hr,margin_ndgc, margin_hr = test(graphrec, device, test_loader, ndgc_k, margin_users)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae),file=outputfile)
        print("ndgc: %.4f, hr:%.4f " % (ndgc, hr), file=outputfile)
        print("margin_ndgc: %.4f, margin_hr:%.4f " % (margin_ndgc, margin_hr), file=outputfile)
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))
        print("ndgc: %.4f, hr:%.4f " % (ndgc, hr))

        if endure_count > 5:
            break

    outputfile.close()

if __name__ == "__main__":
    main()
