import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score



def report(y_true, y_pred):
    # print(y_true)
    # print("-------------------------")
    # print(y_pred)
    # print(len(y_true), len(y_pred))
    # exit()

    TN, FP, FN, TP = get_number(y_true, y_pred)

    tpr = TP/(TP+FN)
    tnr = TN/(TN+FP)

    bacc = (tpr+tnr)/2
    f1 = TP/(TP + (FP+FN)/2)

    auc = roc_auc_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)


    return {'tpr': tpr, 'tnr': tnr, 'bacc': bacc, 'f1': f1, 'auc': auc}



def get_number( y_true, y_pred):
    
    '''TN = sum((y_true==0)&(y_pred==0))
    FP = sum((y_true==0)&(y_pred==1))
    FN = sum((y_true==1)&(y_pred==0))
    TP = sum((y_true==1)&(y_pred==1))'''
    TN, FP, FN, TP = 0, 0, 0, 0

    for i, elm in enumerate(y_pred):
        if (elm == y_true[i]) and (elm == 0):
            TN +=1
        if (elm != y_true[i]) and (elm == 1):
            FP +=1
        if (elm != y_true[i]) and (elm == 0):
            FN +=1
        if (elm == y_true[i]) and (elm == 1):
            TP +=1
        

    return TN, FP, FN, TP