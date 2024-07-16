'''
    This script map cortex structure metric to fsaverage surface using sphere registration
'''

import os
from multiprocessing import Pool

DATA_FOLDER = '../ADNI1/free7_RAS'

SUBJECT_LISTS = os.listdir(DATA_FOLDER)
SUBJECT_LISTS.remove('fsaverage')

os.environ["SUBJECTS_DIR"] = os.path.abspath(DATA_FOLDER)

def process(sub):
    # Files to sample
    hemisphere = ['lh', 'rh']
    files = ['thickness', 'curv', 'sulc']  #------------------------data of MRI------------------------------------
    for hemi in hemisphere:
        for f in files:
            os.system(f'mri_surf2surf --s {sub} --trgsubject fsaverage --hemi {hemi} --sval {DATA_FOLDER}/{sub}/surf/{hemi}.{f} --tval {DATA_FOLDER}/{sub}/surf/{hemi}.{f}.fsaverage.mgh')

with Pool(20) as p:
    print(p.map(process, SUBJECT_LISTS))



'''-------------------------------------------------------------------------------------------
fsaverage is a standard template brain surface model provided by FreeSurfer.
mri_surf2surf:  surface-to-surface transformations on brain surface data. It enables you to resample surface data from one surface space to another, typically using spherical registration methods.
'''