
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import StratifiedKFold
import gc
import random
import os
from utils.utils import seed_everything, train, get_task_data
import argparse
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_absolute_error, balanced_accuracy_score
from utils.data_loader import Feeder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import shutil
# from model import CGModel
from q_pytorch_ADNI1_2modal import AttentionModel        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')       
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')

    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')

    parser.add_argument('--dataset', type=str, choices=['Cora','Citeseer','Pubmed','Chameleon','Squirrel','Actor','Texas','Cornell'],default='Cora')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['GCN', 'APPNP', 'ChebNet','MLP','ChebNetII','ChebBase'], default='ChebNetII')
    parser.add_argument('--prop_lr', type=float, default=0.01, help='learning rate for propagation layer.')
    parser.add_argument('--prop_wd', type=float, default=0.0005, help='learning rate for propagation layer.')

    parser.add_argument('--q', type=int, default=0, help='The constant for ChebBase.')
    parser.add_argument('--full', type=bool, default=False, help='full-supervise with random splits')
    parser.add_argument('--semi_rnd', type=bool, default=False, help='semi-supervised with random splits')
    parser.add_argument('--semi_fix', type=bool, default=False, help='semi-supervised with fixed splits')

    args = parser.parse_args()

    #X: (n, channels, vertices)
    #X, y = np.load('../data/full_ico_data/ADNI1/data.npy').astype('float32'), np.load('../data/full_ico_data/ADNI1/labels.npy')
    X, y = np.load('../volume_student/surface/ADNI1/data_fsaverage_nopvc_ico2.npy').astype('float32'), np.load('../volume_student/surface/ADNI1/labels.npy')    
    
    print(X.shape, y.shape)
    #-------------------------------------------------------------------------------------------------------
    epoch = 150
    batch_size = 8
    seed = 51
    is_cosinelr = False
    criterion = nn.BCELoss()
    lr = 0.001
    
    seed_everything(seed)

    # tasks = ['cn_ad', 'cn_mci']
    # tasks = ['cn_ad', 'cn_mci']
    tasks = ['cn_mci']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_path = './logs/ADNI1_cn_mci_seed{}'.format(seed)

    weight_path = './weights/ADNI1_cn_mci_seed{}'.format(seed)
    '''if os.path.exists('./log_val_pred.txt'):
        os.remove('./log_val_pred.txt')

    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
        os.mkdir(log_path)

    old_path = weight_path.split('/')[0] +'/'+weight_path.split('/')[1] + '/' +weight_path.split('/')[2]
    if os.path.isdir(old_path):
        shutil.rmtree(old_path)
        os.mkdir(old_path)'''
    #-------------------------------------------------------------------------------------------------------


    #from model_q_mod import CGModel
    for i, task in enumerate(tasks):
        print("--------------------------------------------{}-----------------------------------------------------".format(task))
        X_ , y_, = get_task_data(X, y, task)
        print(f"Training data: {X_.shape} {y_.shape}" )
        # exit()
        kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2022)
        fold = 0
        BACC = []
        SEN  = []
        SPE  = []
        AUC_SCORE  = []
        fol_idx = 0
        for train_idx, test_idx in kf.split(X_, y_):

            model = AttentionModel(dims=192,
                               depth=[3,1],
                               heads=3,
                               num_patches=640,
                               num_classes=1,
                               num_channels=4,
                               num_vertices=153,
                               dropout=0.1,
                               branches=[slice(0, 3), slice(3, 4)],
                               activation='sigmoid')

            gc.collect()
            fold += 1
            
            # if fold !=4:
            #     continue
            fold_weight_path = os.path.join(weight_path, task)
            fold_log_path = os.path.join(log_path, task)

            os.makedirs(fold_log_path, exist_ok=True)
            os.makedirs(fold_weight_path, exist_ok=True)

            print(f"*************************************FOLD: {fold}**************************************************")

            X_train, y_train = X_[train_idx], np.expand_dims(y_[train_idx], axis=-1)
            X_test, y_test = X_[test_idx], np.expand_dims(y_[test_idx], axis=-1)
            # print("--------------------")
            # print(X_train.shape, y_train.shape)
            train_feeder = Feeder(X_train, y_train) #data shape: (B, C, P, V) for data Q
            val_feeder = Feeder(X_test, y_test)

            dataloader_train = DataLoader(dataset=train_feeder, batch_size=batch_size, shuffle= True)
            dataloader_val = DataLoader(dataset=val_feeder, batch_size=batch_size, shuffle= False)

            train(model, epoch, device, criterion, fold_weight_path, lr, dataloader_train, \
                  dataloader_val, fol_idx, fold_log_path, is_cosinelr)

            fol_idx +=1
            

            

        



