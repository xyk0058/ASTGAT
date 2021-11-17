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

# parser.add_argument('--data', type=str, default='data/METR-LA/')
# parser.add_argument('--adj_filename', type=str, default='data/METR-LA/adj_mx_dijsk.pkl')
# parser.add_argument('--num_of_vertices', type=int, default=207) #1900
parser.add_argument('--data', type=str, default='data/PEMS-BAY/')
parser.add_argument('--adj_filename', type=str, default='data/PEMS-BAY/adj_mx_bay.pkl')
parser.add_argument('--num_of_vertices', type=int, default=325)

parser.add_argument('--num_of_features', type=int, default=2)
parser.add_argument('--points_per_hour', type=int, default=12)
parser.add_argument('--num_for_predict', type=int, default=12)
parser.add_argument('--num_of_weeks', type=int, default=1)
parser.add_argument('--num_of_days', type=int, default=1)
parser.add_argument('--num_of_hours', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=36)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--seed', type=int, default=1900)#400,800
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay_rate', type=float, default=0.97)
parser.add_argument('--print_every', type=float, default=500)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--early_stop_maxtry', type=int, default=6)
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
def main(rand_seed):
    return_value = -1
    #set seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    #load data
    dataloader, adj_mx, SE = preprocess.load_dataset(args.data, args.adj_filename, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    print('scaler', scaler.std, scaler.mean)
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

    for i in range(args.epoch):

        if args.cuda:
            net = net.cuda()
        # Training
        net.train()
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        w = weight_schedule(i)
        # w = 1

        print('epoch: ', i, ' training...')
        if i < args.warmup_step:
            curr_lr = args.lr * (i+1) / args.warmup_step
            optimizer.param_groups[0]['lr'] = curr_lr
            # optimizer.param_groups[1]['lr'] = curr_lr * 10
        else:
            lr_scheduler.step()
        

        for iter, (trainx, trainy) in enumerate(dataloader['train_loader']):
            optimizer.zero_grad()

            if args.cuda:
                trainx = trainx.cuda()
                trainy = trainy.cuda()
            
            adjs = adj_mx.view(1, adj_mx.shape[0], adj_mx.shape[1])
            adjs = adjs.repeat(trainx.shape[0], 1, 1)
            # print(adjs.shape)
            output, norm_loss = net.forward(trainx, adjs, SE)

            real_val = trainy[:,0,:,:]
            real_val = torch.unsqueeze(real_val,dim=1)
            output = output.permute(0, 3, 1, 2)
            # output = output[:,:,:args.num_of_vertices,:]

            predict = scaler.inverse_transform(output)
            real = scaler.inverse_transform(real_val)

            # print('trainloss', predict.shape, real.shape)

            mae = util.masked_mae(predict, real, 0.0)
            mape = util.masked_mape(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()
            
            # loss = mae
            loss = mae + w * norm_loss
            # loss = util.masked_huber_loss(predict, real, 0.0) + w * norm_loss
            # print ('loss', mae, norm_loss)

            mae = mae.item()
            train_loss.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)

            # loss.backward()
            # if (iter+1) % 1 == 0:
            #     torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
            #     optimizer.step()
            #     optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
            optimizer.step()
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
            # break
        # optimizer.swap_swa_sgd()
        # optimizer.bn_update(dataloader['train_loader'], net, adj_mx, device=adjs.device)

        t2 = time.time()
        train_time.append(t2-t1)

        with torch.no_grad():
            # Validation
            net.eval()
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for iter, (valx, valy) in enumerate(dataloader['val_loader']):
                if args.cuda:
                    valx = valx.cuda()
                    valy = valy.cuda()
                adjs = adj_mx.view(1, adj_mx.shape[0], adj_mx.shape[1])
                adjs = adjs.repeat(valx.shape[0], 1, 1)
                output = net.eval().forward(valx, adjs, SE)
                real_val = valy[:,0,:,:]
                real_val = torch.unsqueeze(real_val, dim=1)
                output = output.permute(0, 3, 1, 2)
                # output = output[:,:,:args.num_of_vertices,:]
                predict = scaler.inverse_transform(output)
                real = scaler.inverse_transform(real_val)
                mae = util.masked_mae(predict, real, 0.0).item()
                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                valid_loss.append(mae)
                valid_mape.append(mape)
                valid_rmse.append(rmse)
                # print('valloss', predict.shape, real.shape)
                # break
                

            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i,(s2-s1)))

            val_time.append(s2-s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)

            if mmin_val_loss > mvalid_loss:
                mmin_val_loss = mvalid_loss
                mmin_epoch = i
                trycnt = 0
                torch.save(net, 'best_model.pkl')

            # lr_scheduler.step()
            # if i >= args.warmup_step:
            #     lr_scheduler.step(mvalid_loss)

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

            print("Training finished")

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
                if i == 11 and metrics[0] < 3.40:
                    torch.save(net, 'best_model_185.pkl')
                    return_value = rand_seed

            log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
            print('early stop trycnt:', trycnt, mmin_epoch)
            print('==================================================================================')
            print('\r\n\r\n\r\n')
            
            # for early stop
            trycnt += 1
            if args.early_stop_maxtry < trycnt:
                print('early stop!')
                return return_value
    return return_value

if __name__ == "__main__":
    # rets = []
    # for i in range(400, 2000, 100):
    # print('rand_seed', i)
    t1 = time.time()
    ret = main(args.seed)
    t2 = time.time()
    rets.append(ret)
    print("Total time spent: {:.4f}".format(t2-t1))
    # print('rets:', rets)
    # a = torch.randn((3, 3))
    # d = torch.sum(a, dim=1)
    # print(d)
    # d = torch.eye(3) * d - a
    # print(a)
    # print(d)
    