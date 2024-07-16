

'''-------------------------------------------------------------------------------------------
fsaverage is a standard template brain surface model provided by FreeSurfer.
mri_surf2surf:  surface-to-surface transformations on brain surface data. It enables you to resample surface data from one surface space to another, typically using spherical registration methods.
'''


import os
import numpy as np
import nibabel as nb
import pandas as pd

def get_subject_name(x):
    x = x.replace('sub-ADNI', '')
    x = x.split('S')
    x = x[0] + '_S_' + x[1]
    return x

def clipping(x):
    return np.clip(x, 0, x.max())

def get_data_fsaverage(sub):
    lh_files = [#---------------all of the below files have 40962 vertices (40962,)-------------
                # MRI features
                'lh.curv.fsaverage.shape.gii',  
                'lh.sulc.fsaverage.shape.gii',
                'lh.thickness.fsaverage.shape.gii',
        
#                  # Amyloid, Tau for ADNI2&3 cohorts             
#                 'lh.amyloid.pvc.fsaverage.shape.gii',            
#                 'lh.tau.pvc.fsaverage.shape.gii',

                # FDG PET for ADNI1 cohorts
                'lh.fdg.nopvc.fsaverage.shape.gii'
    ]
    
    rh_files = [x.replace("lh.", "rh.") for x in lh_files]
    lh_data  = []
    rh_data  = []
    for f in lh_files:
        data = nb.load(f'{DATA_FOLDER}/{sub}/{f}').agg_data()
        if 'amyloid' in f or 'tau' in f or 'fdg' in f:
            data = clipping(data)
        lh_data.append(data)
        
    for f in rh_files:
        data = nb.load(f'{DATA_FOLDER}/{sub}/{f}').agg_data()
        if 'amyloid' in f or 'tau' in f or 'fdg' in f:
            data = clipping(data)
        rh_data.append(data)
        
    lh_data = np.asarray(lh_data)
    rh_data = np.asarray(rh_data)
    
    return [lh_data, rh_data]

# Z-Norm function
def norm(x):
    means = np.asarray([np.mean(x[:, i,...][x[:, i,...] != 0]) for i in range(x.shape[1])  ]).reshape(1, x.shape[1], 1)
    stds  = np.asarray([np.std( x[:, i,...][x[:, i,...] != 0] ) for i in range(x.shape[1])  ]).reshape(1, x.shape[1], 1)
    return (x - means) / stds


'''
    CONFIGURE DATA PATH
'''
DATA_FOLDER = '../data_adni1/'
LABELS      = pd.read_csv('../labels/train_60d_adni1.csv')
OUTPUT_FOLDER = '../data_preprocessed/ADNI1/'
label_ids = {
    'CN' : 0,
    'sMCI':1,
    'pMCI':2,
    'Dementia':3
}

# MAIN PROGRAM
if __name__ == "__main__":
    # Convert text labels to class id
    LABELS['class_id'] = [label_ids[x] for x in LABELS['diagnosis'].values]
    class_id     = LABELS['class_id'].values

    # Fix subject name
    subject_name = [x for x in LABELS['participant_id'].values]
    LABELS['subject_name'] = subject_name


    data_fsaverage = []
    ids  = []
    index = []

    for i, sub in enumerate(subject_name):
        if os.path.exists(f"{DATA_FOLDER}/{sub}") and os.path.exists(f"{DATA_FOLDER}/{sub}/lh.fdg.nopvc.fsaverage.shape.gii"):
            print(f"Preprocessing: {sub}")
            sub_data_fsaverage = get_data_fsaverage(sub)
            data_fsaverage += sub_data_fsaverage
            ids.append(sub)
            index.append(i)

    data_fsaverage = np.asarray(data_fsaverage)


    normalised_data_fsaverage = norm(data_fsaverage)

    # Extract non-overlapping triangular patches
    num_subjects = len(index)
    num_channels = normalised_data_fsaverage.shape[1]
    num_patches  = 320
    num_vertices = 153
    indices_mesh_triangles     = pd.read_csv('../utils/triangle_indices_ico_6_sub_ico_2.csv')

    output_data_fsaverage = np.zeros((num_subjects, num_channels, num_patches * 2, num_vertices))
    for i,id in enumerate(ids):
        print(id)
        for j in range(num_patches):
            indices_to_extract = indices_mesh_triangles[str(j)]

            output_data_fsaverage[i,:,j,:] = normalised_data_fsaverage[2*i][:,indices_to_extract]   #lh_data (4 channels including both feat from mri and fdg)
            output_data_fsaverage[i,:,j + num_patches,:] = normalised_data_fsaverage[2*i + 1][:,indices_to_extract] #rh_data

    class_id = np.asarray(LABELS['class_id'].values[index])
    labels = class_id

    # Save output to npy files
    np.save(f"{OUTPUT_FOLDER}/data_fsaverage_nopvc_ico2.npy", output_data_fsaverage)
    np.save(f"{OUTPUT_FOLDER}/labels_nopvc.npy", labels)





