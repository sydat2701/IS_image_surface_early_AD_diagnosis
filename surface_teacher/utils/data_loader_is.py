import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd 
import os

class Feeder(Dataset):
    def __init__(self, X, Y, data_path, subs):

        self.num_cls = 2
        self.data = X
        self.labels = Y
        self.data_path = data_path
        self.subs = subs

        
    def to_categorical(self, num):
        arr = np.zeros((self.num_cls))
        arr[num-1] = 1
        return torch.from_numpy(arr)
    
    def load_data(self, sub, label):
        #path =os.path.join(self.data_path, sub)
        pet_path = os.path.join(self.data_path, sub, sub+'_fdg_mask_norm_crop.nii.gz') #os.path.join(path, "PET_T1.nii.gz")
        mri_path = os.path.join(self.data_path, sub, sub+'_MRI_mask_norm_crop.nii.gz') #os.path.join(path, "T1.nii.gz")

        pet_img = nib.load(pet_path).get_fdata()
        mri_img = nib.load(mri_path).get_fdata()
        
        # mri_img = self.normalize_data(mri_img)
        #x = [torch.from_numpy(pet_img).unsqueeze(0)]
        x = [torch.from_numpy(mri_img).unsqueeze(0), torch.from_numpy(pet_img).unsqueeze(0)]
        #y = torch.from_numpy(np.array([label]))
        #y = torch.from_numpy(np.array(self.to_categorical(self.label_to_numeric(label))))

        return x
        
    def __len__(self):
        return self.labels.shape[0]


    def __getitem__(self, index):
        #x = torch.from_numpy(self.data[index]).transpose(0,1)
        x = torch.from_numpy(self.data[index])
        y = torch.from_numpy(self.labels[index])

        curr_label = self.labels[index]
        curr_sub = self.subs[index]
        # curr_pet = self.list_all_pet[index]

        imgs = self.load_data(curr_sub, curr_label)

        return [x, imgs], y


