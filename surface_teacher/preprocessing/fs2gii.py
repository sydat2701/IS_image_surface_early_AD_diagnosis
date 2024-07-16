'''
    This script convert freesurfer file format to GIFTI format
'''

import os
from multiprocessing import Pool

original_folder = '../ADNI1/free7_RAS'

data_folder = '../data_adni1'

subject_list = os.listdir(original_folder)

subject_list.remove('fsaverage')

files = [
        #   "lh.thickness.fsaverage.mgh", "rh.thickness.fsaverage.mgh",
        #   "lh.curv.fsaverage.mgh", "rh.curv.fsaverage.mgh",
        #   "lh.sulc.fsaverage.mgh", "rh.sulc.fsaverage.mgh",

        ## ADNI1 Cohorts
        #    "lh.fdg.nopvc.fsaverage.nii.gz", "rh.fdg.nopvc.fsaverage.nii.gz",

        ## ADNI2 & ADNI3 Cohorts
        #   'lh.amyloid.pvc.fsaverage.nii.gz', 'rh.amyloid.pvc.fsaverage.nii.gz',
        #  'lh.tau.pvc.fsaverage.nii.gz',  'rh.tau.pvc.fsaverage.nii.gz',
         ]

error_sub = []

def process(sub):
    if not os.path.exists(f'{data_folder}/{sub}'):
        os.makedirs(f'{data_folder}/{sub}')
    
    for f in files:
        f_ = f
        input_file = f'{original_folder}/{sub}/surf/{f_}'

        if not os.path.exists(input_file):
            # error_sub.append(sub)
            print(input_file)
            return False

        outfile = f_.replace('.nii.gz', '')
        outfile = outfile.replace('.mgh', '')
        if 'sphere' in outfile:
            outfile = f'{outfile}.surf.gii'
        else:
            outfile = f'{outfile}.gii'

        output_file = f'{data_folder}/{sub}/{outfile}'
        
        if 'sphere' not in f_:
            os.system(f'mri_convert {input_file} {output_file} > /dev/null')
        else:
            os.system(f'mris_convert {input_file} {output_file} > /dev/null')
    return True

with Pool(10) as p:
    print(p.map(process, subject_list))

