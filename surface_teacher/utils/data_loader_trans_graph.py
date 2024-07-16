import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd 

class Feeder(Dataset):
    def __init__(self, X, Y, X_ico):

        self.num_cls = 2
        self.data = X
        self.labels = Y
        self.data_ico = X_ico

        
    def to_categorical(self, num):
        arr = np.zeros((self.num_cls))
        arr[num-1] = 1
        return torch.from_numpy(arr)
        
    def __len__(self):
        return self.labels.shape[0]


    def __getitem__(self, index):
        #x = torch.from_numpy(self.data[index]).transpose(0,1)
        x = torch.from_numpy(self.data[index])
        y = torch.from_numpy(self.labels[index])
        x_ico = torch.from_numpy(self.data_ico[index])
        return [x, x_ico], y


