from __future__ import print_function
import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import torch.utils.data as data
import util
import gc


class NetDataSet(data.Dataset):
    def __init__(self, x, y, batch_size, edge_num, adj_mx, isTrain, pad_with_last_sample=True):
        self.edge_num = edge_num
        self.adj_mx = adj_mx
        self.isTrain = isTrain
        self.x = x.astype('float32')
        self.y = y.astype('float32')
        self.nodes_num = self.adj_mx.shape[0]
        if pad_with_last_sample:
            num_padding = (batch_size - (len(self.x) % batch_size)) % batch_size
            x_padding = np.repeat(self.x[-1:], num_padding, axis=0)
            y_padding = np.repeat(self.y[-1:], num_padding, axis=0)
            self.x = np.concatenate([self.x, x_padding], axis=0)
            self.y = np.concatenate([self.y, y_padding], axis=0)

    def __getitem__(self, index):
        x_ = self.x[index].transpose(1, 0, 2)
        # x_zeros = np.zeros((self.edge_num, x_.shape[1], x_.shape[2]))
        # x_zeros = np.ones((self.edge_num, x_.shape[1], x_.shape[2]))

        # idx = 0
        # for i in range(self.nodes_num):
        #     for j in range(self.nodes_num):
        #         if self.adj_mx[i][j] > 0:
        #             x_zeros[idx] = x_[i] + x_[j] / 2
        #             idx += 1

        # x_ = np.concatenate((x_, x_zeros), axis=0).astype('float32')
        y_ = self.y[index].transpose(2, 1, 0)
        # print('x_y_', x_.shape, y_.shape)
        return x_, y_
    
    def __len__(self):
        return self.x.shape[0]
    
    def get_dataset_type():
        return self.dataset


def DataLoader(x, y, batch_size, edge_num, adj_mx, shuffle=True, isTrain=False):
    dataset = NetDataSet(x, y, batch_size, edge_num, adj_mx, isTrain)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_pickle(pkl_filename)
    return adj_mx

def load_se(se_filename):
    f = open(se_filename, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1 :]
    return SE

def gen_edge_node_graph(origon_adj, data):
    origon_node_num = origon_adj.shape[0]
    edge_num = np.sum((origon_adj>0).astype(int))
    new_node_num = origon_node_num + edge_num
    new_adj_mx = np.zeros((new_node_num, new_node_num), int)
    edge_idx = origon_node_num
    for i in range(origon_node_num):
        for j in range(origon_node_num):
            if origon_adj[i][j] > 0:
                new_adj_mx[i][edge_idx] = 1
                new_adj_mx[edge_idx][j] = 1
                edge_idx += 1
    print('edge_idx', edge_idx)
    return new_adj_mx, edge_num



def load_dataset(dataset_dir, adj_filename, batch_size, valid_batch_size= None, test_batch_size=None):
    SE = load_se(dataset_dir+'/SE.txt')
    print('SE', SE.shape)
    adj_mx = load_adj(adj_filename)
    print('adj', adj_mx.shape)
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x'].astype('float32')
        data['y_' + category] = cat_data['y'].astype('float32')
    cat_data = []
    # (36465, 12, 325, 2)
    # print('zero num:', data['x_train'].shape, sum(data['x_train']==0))
    # for batch in range(data['x_train'].shape[0]):
    #     for t in range(data['x_train'].shape[1]):
    #         for node in range(data['x_train'].shape[2]):
    #             for f in range(data['x_train'].shape[3]):
    #                 if data['x_train'][batch][t][node][f] == 0 and t > 0 and t < data['x_train'].shape[1]-1:
    #                     # data['x_train'][batch][t][node][f] = data['x_train'][batch][t-1][node][f] + data['x_train'][batch][t+1][node][f]
    #                     print('zero')
    # print('x_train', data['x_train'].shape, data['y_train'])
    # edge_adj_mx, edge_num = gen_edge_node_graph(adj_mx, data)
    

    edge_num = 0
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    
    
    print('dataloader')
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, edge_num, adj_mx, isTrain=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size, edge_num, adj_mx, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, edge_num, adj_mx, shuffle=False)
    data['scaler'] = scaler
    print('train shape', data['x_train'].shape, data['y_train'].shape)
    print('val shape', data['x_val'].shape, data['y_val'].shape)
    print('test shape', data['x_test'].shape, data['y_test'].shape)
    print('preprocess finished.')
    return data, [adj_mx], SE

def dataInterpolation(x):
    # (23974, 12, 207, 2) (23974, 12, 207, 2)
    x_mean = np.mean(x, axis=1, keepdims=True)
    print('mean', x_mean.shape, x.shape)
    cnt = 0
    sample = x.shape[0]
    timestep = x.shape[1]
    nodes = x.shape[2]
    for s in range(sample):
        for n in range(nodes):
            for t in range(timestep):
                if x[s,t,n,0] == 0:
                    cnt += 1
                    x[s,t,n,0] = x_mean[s,0,n,0]
    print('0 cnt', cnt)
    return x