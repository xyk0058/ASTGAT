import time
import util
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from model.stgat2 import STGAT, STGATModel
from loss.MSELoss import mse_loss
from loss.MAPELoss import MAPELoss

parser = argparse.ArgumentParser()
# parser.add_argument('--graph_signal_matrix_filename', type=str, default='data/METR-LA/data2.npz')
# parser.add_argument('--data', type=str, default='data/METR-LA/')
parser.add_argument('--data', type=str, default='data/PEMS-BAY/')
# parser.add_argument('--adj_filename', type=str, default='data/METR-LA/adj_mx_dijsk.pkl')
parser.add_argument('--adj_filename', type=str, default='data/PEMS-BAY/adj_mx_bay.pkl')
# parser.add_argument('--params_dir', type=str, default='experiment_METR_LA')
# parser.add_argument('--num_of_vertices', type=int, default=207)
parser.add_argument('--num_of_vertices', type=int, default=325)
parser.add_argument('--num_of_features', type=int, default=2)
parser.add_argument('--points_per_hour', type=int, default=12)
parser.add_argument('--num_for_predict', type=int, default=12)
parser.add_argument('--num_of_weeks', type=int, default=1)
parser.add_argument('--num_of_days', type=int, default=1)
parser.add_argument('--num_of_hours', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--print_every', type=float, default=200)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--adjtype', type=str, default='symnadj')
parser.add_argument('--early_stop_maxtry', type=int, default=1000)
parser.add_argument('--cuda', action='store_true', help='use CUDA training.')

args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
print(f'Training configs: {args}')


def weight_schedule(epoch, max_val=10, mult=-5, max_epochs=100):
    if epoch == 0:
        return 0.
    w = max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)
    w = float(w)
    if epoch > max_epochs:
        return max_val
    return w

def main():
    #set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #load data
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adj_filename, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    

    scaler = dataloader['scaler']
    
    adj_mx = torch.from_numpy(np.array(adj_mx))[0]
    adj_mx_ = torch.from_numpy(np.random.permutation(np.array(adj_mx)))[0]
    
    if args.cuda:
        adj_mx = adj_mx.cuda()
        adj_mx_ = adj_mx_.cuda()

    print('adj', adj_mx.shape)

    net = STGATModel(args.cuda, args.num_of_vertices, args.num_of_features, args.points_per_hour*args.num_of_hours, args.num_for_predict)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-8)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, min_lr=1e-7, eps=1e-08)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    mmin_val_loss = 10000000
    mmin_val_loss_1 = 10000000
    mmin_val_loss_2 = 10000000
    mmin_epoch = 10000000
    trycnt = 0

    for i in range(args.epoch):
        if args.cuda:
            net = net.cuda()
        # # Training
        net.train()
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        w2 = weight_schedule(i)
        for iter, (trainx, trainy) in enumerate(dataloader['train_loader']):
            optimizer.zero_grad()

            if args.cuda:
                trainx = trainx.cuda()
                trainy = trainy.cuda()
            
            output = net.forward(trainx, adj_mx)
            # output2 = net.forward(trainx, adj_mx)

            real_val = trainy[:,0,:,:]
            real_val = torch.unsqueeze(real_val,dim=1)
            output = output.permute(0, 3, 1, 2)

            # output2 = output2.permute(0, 3, 1, 2)

            predict = scaler.inverse_transform(output)
            real = scaler.inverse_transform(real_val)
            # print('loss shape', real.shape, predict.shape)
            mae = util.masked_mae(predict, real, 0.0)
            mape = util.masked_mape(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()
            
            loss = mae
            # predict2 = scaler.inverse_transform(output2)
            # mae2 = util.masked_mae(predict, predict2, 0.0)
            # loss = mae + mae2 * w2

            mae = mae.item()
            train_loss.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
                # print('mae, mae2', mae, mae2)
            break
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
                output = net.eval().forward(valx, adj_mx)
                real_val = valy[:,0,:,:]
                real_val = torch.unsqueeze(real_val, dim=1)
                output = output.permute(0, 3, 1, 2)
                predict = scaler.inverse_transform(output)
                real = scaler.inverse_transform(real_val)
                mae = util.masked_mae(predict, real, 0.0).item()
                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                valid_loss.append(mae)
                valid_mape.append(mape)
                valid_rmse.append(rmse)
                break

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

            # lr_scheduler.step()
            lr_scheduler.step(mvalid_loss)

            # Testing
            outputs = []
            realy = []
            for iter, (testx, testy) in enumerate(dataloader['test_loader']):
                if args.cuda:
                    testx = testx.cuda()
                    testy = testy.cuda()
                
                output = net.forward(testx, adj_mx)

                output = output.permute(0, 3, 1, 2)
                output = output.squeeze()
                outputs.append(output)
                realy.append(testy[:,0,:,:].squeeze())
                print('loss', output.shape, testy[:,0,:,:].squeeze().shape)

            yhat = torch.cat(outputs, dim=0)
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

            log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
            print('early stop trycnt:', trycnt, mmin_epoch)
            print('==================================================================================')
            print('\r\n\r\n\r\n')
            
            # for early stop
            trycnt += 1
            if args.early_stop_maxtry < trycnt:
                print('early stop!')
                return


    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
