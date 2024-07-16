'''
    This script resample all metrics file to 6th-order icosphere using human connectome workbench
    wb (command): workbench

'''

import os
from multiprocessing import Pool

data_folder = '../data_adni1_v2'

subject_list = os.listdir(data_folder)

def process(sub):
    print(sub)
    files = [#---------------all of the below files have 163842 vertices (163842,)-------------
            ## Fsaverage template space
             "lh.thickness.roi.fsaverage", "rh.thickness.roi.fsaverage", 
             "lh.curv.roi.fsaverage", "rh.curv.roi.fsaverage",
             "lh.sulc.roi.fsaverage", "rh.sulc.roi.fsaverage",
#              'lh.amyloid.pvc.roi.fsaverage', 'rh.amyloid.pvc.roi.fsaverage',
#              'lh.tau.pvc.roi.fsaverage', 'rh.tau.pvc.roi.fsaverage',
             'lh.fdg.nopvc.roi.fsaverage', 'rh.fdg.nopvc.roi.fsaverage',
             ]

    for f in files:
        input_file = f'{data_folder}/{sub}/{f}.gii' #-----file of features that want to resample
        output_file = f'{data_folder}/{sub}/{f}.shape.gii' #output shape is (40962,)
        
        if 'lh' in f:
            input_surf = '../Icospheres/lh.average.surf.gii' #-----surface of the original feature
            if 'fsaverage' not in f:
                input_surf = f'{data_folder}/{sub}/lh.sphere.surf.gii'

            target_surf = '../Icospheres/gtmseg'    #-----the new surface that want to resample feature to
            os.system(f'wb_command -metric-resample {input_file} {input_surf} {target_surf} BARYCENTRIC {output_file}')
        else:
            input_surf = '../Icospheres/rh.average.surf.gii'
            if 'fsaverage' not in f:
                input_surf = f'{data_folder}/{sub}/rh.sphere.surf.gii'

            target_surf = '../Icospheres/ico-6.R.surf.gii' #(40962, 3)
            os.system(f'wb_command -metric-resample {input_file} {input_surf} {target_surf} BARYCENTRIC {output_file}')
            os.system(f'wb_command -set-structure {output_file} CORTEX_LEFT')
    
for sub in subject_list:
    process(sub)


'''self-wb command:
wb_command -metric-resample "./rh.curv.roi.fsaverage.gii" "../../../process_surface_data/Icospheres/rh.average.surf.gii" "../../../process_surface_data/Icospheres/ico-6.R.surf.gii" BARYCENTRIC "./tmp/rh.curv.roi.fsaverage.shape.gii"
'''