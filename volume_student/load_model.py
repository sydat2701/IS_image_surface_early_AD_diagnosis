from teacher.data_loader_is import Feeder
from torch.utils.data import DataLoader
import os
import glob
import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import numpy as np
from eval_metrics import report

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

def get_subs_labels(subs, y, task):
    neg_idx = pos_idx = -1
    sub_return = []
    y_return = []
    if task == 'cn_ad':
        neg_idx = 0
        pos_idx = 3
    elif task == 'cn_mci':
        neg_idx = 0
        pos_idx = (1, 2)


    for i, class_id in enumerate(y):
        if class_id == neg_idx:
            sub_return.append(subs[i])
            y_return.append(0)
        
        if isinstance(pos_idx, tuple):
            if class_id in pos_idx:
                sub_return.append(subs[i])
                y_return.append(1)
                
        elif class_id == pos_idx:
            sub_return.append(subs[i])
            y_return.append(1)
            
    return np.asarray(sub_return), np.asarray(y_return)


def get_model(path, fold_idx, teacher_weight_path):
    
    model = MVNet(teacher_weight_path)
    weight_path = glob.glob(path+'/*'+str(fold_idx)+'.pth')

    state_dict = torch.load(weight_path[0], map_location='cpu')
    first_key, first_value = list(state_dict.items())[0]

    if first_key.split('.')[0] == 'module': #use data parallel
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}    
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    

    return model


# path = '/home/id202388544/dat/MVNet_fol/MVNet_surfteacher/teacher/weights/final_fdg_surf'
path = '/home/id202388544/dat/MVNet_fol/MVNet_surfteacher/teacher/weights/fdg_surf'
teacher_path = '/home/id202388544/dat/MVNet_fol/MVNet_surfteacher/teacher/teacher_weights/cn_mci'

for i in range(2,4):
    fol_to_load = i
    teacher_weight_path = os.path.join(teacher_path, glob.glob(teacher_path+'/*fol'+str(fol_to_load)+'.pth')[0])

    model = get_model(path, fol_to_load, teacher_weight_path)
    print("load succesfully fol {}!".format(i))
    # print(model)
    model.eval()

    del model.extract_net.surt_teacher
    '''print("--------------------------------------")
    # print(model)
    x = torch.rand((1, 1, 160, 192, 160))

    res = model(x)
    print("act res: ", res)
    # print(res.shape)
    # for name, module in model.named_modules():
    #     print(name)
    torch.save(model, 'weights.pth')
    
    new_mod = torch.load("weights.pth")
    new_res = new_mod(x)
    print("new res: ", new_res)'''

    data_path = '/home/id202388544/dat/Data_ADNI1_ANTS'
    X, y = np.load('/home/id202388544/dat/surface/data/ADNI1/data_fsaverage_nopvc_ico2.npy').astype('float32'), np.load('/home/id202388544/dat/surface/data/ADNI1/labels.npy')
    subs = np.load('./teacher/surf_ids/sub.npy', allow_pickle=True)
    ids = np.load('./teacher/surf_ids/labels.npy', allow_pickle=True)
    X_ , y_, = get_task_data(X, y, 'cn_mci')
    kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2022)
    list_subs, list_y = get_subs_labels(subs, ids, 'cn_mci')
    model.cuda()
    fol_idx = 0
    for train_idx, test_idx in kf.split(X_, y_):
        if fol_idx not in [0,]:
            fol_idx +=1
            continue

        subs_train = list_subs[train_idx]
        subs_test = list_subs[test_idx]

        X_train, y_train = X_[train_idx], np.expand_dims(y_[train_idx], axis=-1)
        X_test, y_test = X_[test_idx], np.expand_dims(y_[test_idx], axis=-1)
        # print("--------------------")
        # print(X_train.shape, y_train.shape)
        #train_feeder = Feeder(X_train, y_train, data_path, subs_train) #data shape: (B, C, P, V) for data Q
        val_feeder = Feeder(X_test, y_test, data_path, subs_test)
        #print(X_test.shape, y_test.shape, subs_test.shape)
        dataloader_validation = DataLoader(dataset=val_feeder, batch_size=4, shuffle= False)
        #print(dataloader_validation)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        list_labels = []
        list_preds = []
        num_iters_val = 0
        epoch_val_acc = 0
        epoch_val_loss = 0
        for ii, batch in enumerate(dataloader_validation):
            imgs = batch[0][1][1].to(device).float() #fdg
            x = batch[0][0].to(device).float()
            labels = batch[1].to(device).float()

            num_iters_val += 1
            with torch.no_grad():
                y_pred = model(imgs)
            #val_acc = ((labels.argmax(1, keepdim=True)).eq(y_pred.argmax(1, keepdim=True)).sum()).float()/labels.shape[0]
            #tmp= (labels.view(32,-1))
            #val_acc = (tmp.eq(y_pred.argmax(1, keepdim=True)).sum()).float()/tmp.shape[0]
            #------------------------------------------------------------------
            # ps = torch.exp(y_pred).data
            # y_pred = torch.where(y_pred > 0.5, torch.tensor(1), torch.tensor(0))
            y_pred = torch.round(y_pred)
            #print("here 1.5")
            equality = (labels.data == y_pred.data)
            #print("here 1.6")
            val_acc = equality.type_as(torch.FloatTensor()).mean()
            #-------------------------------------------------------------------
            # print(labels)
            # print(y_pred)
            
            
            
            list_labels.extend(labels.cpu().detach().numpy().reshape(-1).tolist())
            list_preds.extend(y_pred.cpu().detach().numpy().reshape(-1).tolist())

            epoch_val_acc +=val_acc.item()
        val_accuracy = epoch_val_acc / (num_iters_val)
        val_loss = epoch_val_loss / num_iters_val

        report_dict =  report(list_labels, list_preds)
        
        epoch_val_loss =0
        epoch_val_acc=0
        num_iters_val=0
        print("val_acc:", val_accuracy, "val_loss:", val_loss)
        print("Report in detail: ", report_dict)
    
        fol_idx +=1



    # break



















