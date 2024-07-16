"""
    This script run FreeSurfer and PetSurfer to all subjects
"""

import os
import shutil
import numpy as np
import glob
from multiprocessing import Pool
import pandas as pd

RAW_DATA_FOLDER = '../ADNI1_RAW/Nifti'
DATA_FOLDER = '../ADNI1/free7_RAS'

SUBJECT_LISTS = os.listdir('../ADNI1/free7_RAS')
SUBJECT_LISTS.remove('fsaverage')

os.environ["SUBJECTS_DIR"] = os.path.abspath(DATA_FOLDER)

def process(sub):
    print(f"Processing: {sub}")
    subject_dir = f'{DATA_FOLDER}/{sub}'
    if not os.path.exists(subject_dir):
        return False
        
    FWHM = 8

    ###
    #   Run recon-all and gtmseg (FreeSurfer)
    ###
    mri_file = glob.glob(f'{RAW_DATA_FOLDER}/{sub}/*MPRAGE*.nii')
    os.system(f"recon-all -all -s {sub} -i {mri_file}> /dev/null")
    os.system(f"gtmseg --s {sub} > /dev/null")  #------------------------- segment gray matter---------------------------
    
    ###
    #   PetSurfer
    ###
    pet_folder = f'{subject_dir}/pet_uniform/FDG'

    pet_file = glob.glob(f'{RAW_DATA_FOLDER}/{sub}/*FDG*.nii')

    if len(pet_file) == 0:
        print(f"{sub}")
        return False

    pet_file = pet_file[0]

    # # Create average template.nii.gz
    os.system(f"mri_concat {pet_file} --mean --o {pet_folder}/template.nii.gz > /dev/null")

    # # Register to anatomical space
    os.system(f"mri_coreg --s {sub} --mov {pet_folder}/template.nii.gz --reg {pet_folder}/template.reg.lta > /dev/null")

    # # SAMPLE ONTO FSAVERAGE SURFACE #---------------transform 3d image to surface------------------------------
    pet_folder = f'{subject_dir}/pet_uniform/FDG'
    os.system(f"mri_vol2surf --mov {pet_folder}/PET_T1.nii.gz --regheader {sub} --hemi lh --projfrac 0.5 --o {subject_dir}/surf/lh.fdg.nopvc.fsaverage.nii.gz --cortex --surf white --trgsubject fsaverage")
    os.system(f"mri_vol2surf --mov {pet_folder}/PET_T1.nii.gz --regheader {sub} --hemi rh --projfrac 0.5 --o {subject_dir}/surf/rh.fdg.nopvc.fsaverage.nii.gz --cortex --surf white --trgsubject fsaverage")

    print(f"Done processing {sub}")

    return True

with Pool(10) as p:
    print(p.map(process, SUBJECT_LISTS))