import os

import torch
from torch._C import set_flush_denormal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from utils.eval_metrics import report
import random
import copy
import nibabel as nib

from tqdm import tqdm

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    lr = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)*0.8
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Current lr: ", lr)

def cosine_scheduler(optimizer, initial_lr, min_lr, epochs_per_cycle, epoch):
    cycle = 50
    
    lr = min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * (epoch % cycle) / cycle)) / 2
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Current lr: ", lr)


'''
    DATA UTILITY
'''

def get_task_data(X, y, task):
    neg_idx = pos_idx = -1
    X_return = []
    y_return = []
    if task == 'cn_ad':
        neg_idx = 0
        pos_idx = 3
    elif task == 'cn_mci':
        neg_idx = 0
        pos_idx = (1, 2)


    for i, class_id in enumerate(y):
        if class_id == neg_idx:
            X_return.append(X[i])
            y_return.append(0)
        
        if isinstance(pos_idx, tuple):
            if class_id in pos_idx:
                X_return.append(X[i])
                y_return.append(1)
                
        elif class_id == pos_idx:
            X_return.append(X[i])
            y_return.append(1)
            
    return np.asarray(X_return), np.asarray(y_return)



def train(model, num_epoch, device, criterion, weight_path, lr, dataloader_train, dataloader_val, \
          fol_idx, log_path, is_cosinelr):
    
    global latest_saved_weight
    latest_saved_weight = None

    model.to(device)
    beta = gamma = 0.5
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.00005) #0.00005
    # sched = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
    if is_cosinelr:
        # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 148//batch_size)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20, eta_min = 1e-5)
        # sched = CosineAnnealingWarmupRestarts(optimizer,
        #                                   first_cycle_steps=20,
        #                                   cycle_mult=1.0,
        #                                   max_lr=0.0003,
        #                                   min_lr=1e-7,
        #                                   warmup_steps=0,
        #                                   gamma=0.8)
    
    #best_model_weights = copy.deepcopy(model.state_dict())
    best_val_loss= 99999999
    best_val_acc = -99999999
    best_val_bacc = -99999999
    num100 = 0
    for epoch in range(num_epoch):
        print("Epoch: {}".format(epoch+1))
        epoch_train_acc = 0
        epoch_train_loss=0
        epoch_loss_genx = 0
        epoch_loss_geny = 0
        num_iters_train=0
        model.train()
        
        #adjust_learning_rate(optimizer, epoch, num_epoch, lr)
        cosine_scheduler(optimizer, lr, min_lr=1e-5, epochs_per_cycle=num_epoch, epoch= epoch)
        pbar_train = tqdm(dataloader_train)
        for inputs, labels in pbar_train:
            #labels.squeeze(1)
            optimizer.zero_grad()

            x = inputs.to(device).float()
            labels = labels.to(device).float()

            num_iters_train += 1
            #print("here 0")
            
            y_pred = model(x)
            # print(">>>>>>>>>>>>>> ", y_pred.shape, labels.shape)
            # print(labels)

            loss= criterion(y_pred, labels)

            epoch_train_loss += loss

            loss.backward()
            optimizer.step()

            #optimizer.zero_grad()
            # sched.step()

            y_pred = torch.round(y_pred)
            # print(y_pred.item())
            #------------------------------------------------------------------
            equality = (labels.data == y_pred.data)
            train_acc = equality.type_as(torch.FloatTensor()).mean()
            #-------------------------------------------------------------------
            epoch_train_acc +=train_acc
            #tqdm.write(f"Loss: {loss:.4f}, acc: {train_acc:.4f}", end="\r")
            pbar_train.set_description(f"Loss: {epoch_train_loss.item()/num_iters_train:.4f}, acc: {epoch_train_acc.item()/num_iters_train:.4f}")


        train_accuracy = epoch_train_acc / (num_iters_train)
        train_loss = epoch_train_loss / num_iters_train
        
        if np.round(train_accuracy.cpu().numpy(), 2)==1.0:
            num100 +=1
        
        if is_cosinelr:
            sched.step()
            for param_group in optimizer.param_groups:
                print("Current lr: ", param_group['lr'])
        
        log_train = open(os.path.join(log_path, 'train.txt'), 'a')
        log_train.write('Epoch {}: train_acc {}, train_loss {}\n'.format(epoch, train_accuracy.item(), \
                                                                       (train_loss/num_iters_train).item() ))
        log_train.close()

        epoch_train_loss =0
        epoch_train_acc=0
        num_iters_train=0
        #print("epoch", (epoch+1))
        print("train_acc:", train_accuracy.item(), "train_loss:", train_loss.item())

        #---------------------------------------VALIDATION---------------------------
        epoch_val_acc = 0
        epoch_val_loss=0
        num_iters_val = 0

        list_labels = []
        list_preds = []

        model.eval()
        f = open('log_val_pred.txt', 'a')
        if epoch == 0:
            f.write("######################################{}############################################\n".format(fol_idx))
        for inputs, labels in tqdm(dataloader_val):
            x = inputs.to(device).float()
            labels = labels.to(device).float()

            num_iters_val += 1
            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, labels)

            epoch_val_loss += loss

            #------------------------------------------------------------------
            # ps = torch.exp(y_pred).data
            # y_pred = torch.where(y_pred > 0.5, torch.tensor(1), torch.tensor(0))
            y_pred = torch.round(y_pred)
            #print("here 1.5")
            equality = (labels.data == y_pred.data)
            #print("here 1.6")
            val_acc = equality.type_as(torch.FloatTensor()).mean()
            #-------------------------------------------------------------------           
            f.write(str(labels.cpu().detach().numpy())+'\n')
            f.write(str(y_pred.cpu().detach().numpy())+'\n')
            f.write("---------------------------------------------\n")
            
            list_labels.extend(labels.cpu().detach().numpy().reshape(-1).tolist())
            list_preds.extend(y_pred.cpu().detach().numpy().reshape(-1).tolist())

            epoch_val_acc +=val_acc
        val_accuracy = epoch_val_acc / (num_iters_val)
        val_loss = epoch_val_loss / num_iters_val

        report_dict =  report(list_labels, list_preds)
        
        epoch_val_loss =0
        epoch_val_acc=0
        num_iters_val=0
        f.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        f.close()
        print("val_acc:", val_accuracy.item(), "val_loss:", val_loss.item())
        print("Report in detail: ", report_dict)


        log_val = open(os.path.join(log_path, 'val.txt'), 'a')
        log_val.write('Epoch {}: val_acc {}, val_loss {}\n'.format(epoch, val_accuracy.item(), \
                                                                       val_loss.item()))
        log_val.close()

        sen, spe, bacc, auc = report_dict['tpr'], report_dict['tnr'], report_dict['bacc'], report_dict['auc']
        # if bacc> best_val_bacc:
        #     best_val_bacc = bacc
        #     print("--------------------------best bacc-------------------------------------")
        
        if bacc> best_val_bacc:
        # if val_loss<best_val_loss:
            best_val_bacc = bacc
            #best_val_loss = val_loss

            if latest_saved_weight !=None:
                os.remove(latest_saved_weight)
            tmp_path = weight_path+'/{:.4f}acc_{:.4f}bacc_{:.4f}auc_{:.4f}sen_{:.4f}spe'.format( \
                val_accuracy, bacc, auc, sen, spe)+ '_fol'+str(fol_idx)+'.pth'
            latest_saved_weight = tmp_path
            torch.save(model.state_dict(), tmp_path)
            print("----------------------->saved model<--------------------")
        
        # if num100 >= 30:
        #     break