import time
import util
import preprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from model.model import NEGAT
from loss.MSELoss import mse_loss
from loss.MAPELoss import MAPELoss
import gc
import sys
from optimizer.RAdam import RAdam
from optimizer.SWA import SWA

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
# parser.add_argument('--graph_signal_matrix_filename', type=str, default='data/METR-LA/data2.npz')

parser.add_argument('--data', type=str, default='data/METR-LA/')
parser.add_argument('--adj_filename', type=str, default='data/METR-LA/adj_mx_dijsk.pkl')
parser.add_argument('--num_of_vertices', type=int, default=207)
# parser.add_argument('--data', type=str, default='data/PEMS-BAY/')
# parser.add_argument('--adj_filename', type=str, default='data/PEMS-BAY/adj_mx_bay.pkl')
# parser.add_argument('--num_of_vertices', type=int, default=325)

parser.add_argument('--num_of_features', type=int, default=2)
parser.add_argument('--points_per_hour', type=int, default=12)
parser.add_argument('--num_for_predict', type=int, default=12)
parser.add_argument('--num_of_weeks', type=int, default=1)
parser.add_argument('--num_of_days', type=int, default=1)
parser.add_argument('--num_of_hours', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--lr_decay_rate', type=float, default=0.97)
parser.add_argument('--print_every', type=float, default=500)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--early_stop_maxtry', type=int, default=150)
parser.add_argument('--cuda', action='store_true', help='use CUDA training.')
parser.add_argument('--warmup_step', type=int, default=5)
parser.add_argument('--T_max', type=int, default=32)

args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
print(f'Training configs: {args}')


def weight_schedule(epoch, max_val=0.1, mult=-5, max_epochs=30):
    if epoch == 0:
        return 0.
    w = max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)
    w = float(w)
    if epoch > max_epochs:
        return max_val
    return w

def get_dataloader(dataloader_type, batch_size):
    x_ = np.load('x_' + dataloader_type + '.npy')
    y_ = np.load('y_' + dataloader_type + '.npy')
    dataloader = preprocess.DataLoader(x_, y_, batch_size)
    return dataloader

# @profile(precision=4, stream=open('memory_profiler.log','w+'))
def main():
    #set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #load data
    dataloader, adj_mx, SE = preprocess.load_dataset(args.data, args.adj_filename, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    adj_mx = torch.from_numpy(np.array(adj_mx))[0]
    adj_mx = adj_mx.type(torch.FloatTensor)

    SE = torch.from_numpy(SE)
    print('SE shape', SE.shape)

    # adj_mx = torch.ones_like(adj_mx)

    if args.cuda:
        adj_mx = adj_mx.cuda()
        SE = SE.cuda()
    
    net = NEGAT(args.cuda, adj_mx, adj_mx.shape[0], args.num_of_features, args.points_per_hour*args.num_of_hours, args.
    num_for_predict)
    if args.cuda:
        net = net.cuda()
    

    generator_params = list(map(id, net.network_generator.parameters()))
    base_params = filter(lambda p: id(p) not in generator_params,
                     net.parameters())
    # optimizer = torch.optim.Adam([{'params': base_params}, {'params': net.network_generator.parameters(), 'lr': args.lr * 10}], lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-9)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-9)
    # optimizer = RAdam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-9)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: args.lr_decay_rate ** epoch)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_max, eta_min=0, last_epoch=-1)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, min_lr=1e-7, eps=1e-09)
    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    trycnt = 0
    mmin_val_loss = 10000000
    mmin_val_loss_1 = 10000000
    mmin_val_loss_2 = 10000000
    mmin_epoch = 10000000
    train_step = 0

    net = torch.load('bestModel/best_model_metrla.pkl')

    if args.cuda:
        net = net.cuda()
    with torch.no_grad():
        net.eval()
        # Testing
        outputs = []
        realy = []
        for iter, (testx, testy) in enumerate(dataloader['test_loader']):
            if args.cuda:
                testx = testx.cuda()
                testy = testy.cuda()
            
            adjs = adj_mx.view(1, adj_mx.shape[0], adj_mx.shape[1])
            adjs = adjs.repeat(testx.shape[0], 1, 1)
            output = net.forward(testx, adjs, SE)

            output = output.permute(0, 3, 1, 2)
            output = output.squeeze()
            outputs.append(output)
            realy.append(testy[:,0,:,:].squeeze())
            # print('testloss', testy[:,0,:,:].squeeze().shape, output.shape)


        yhat = torch.cat(outputs, dim=0)
        outputs = []
        realy = torch.cat(realy, dim=0)
        if args.cuda:
            yhat = yhat.cuda()
            realy = realy.cuda()


        amae = []
        amape = []
        armse = []
        for i in range(12):
            pred = scaler.inverse_transform(yhat[:,:,i])
            real = scaler.inverse_transform(realy[:,:,i])
            metrics = util.metric(pred,real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
        print('==================================================================================')
        print('\r\n\r\n\r\n')


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
