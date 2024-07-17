1. This code is for review process. The final code will be updated. Because of the ADNI dataset policy, we can not public the preprocessed data. You can access this dataset from the official website.
2. The surface data after preprocessing should be located at 'volume_student/surface'. Their respective subjectID and labels (ground truths) are located at 'volume_student/teacher/surf_ids'. The pretrained weights of surface teacher should be located at 'volume_student/teacher/teacher_weights' for training volume student. 
3. Examples of pretrained weights for surface teacher on ADNI23 can be obtained from this link https://drive.google.com/file/d/14rFdonCqQQpq7cHNwtnZTirkTWLkMqRG/view?usp=sharing.
4. Directory tree for processed ADNI volume (e.g ADNI1):
Data_ADNI1 \\
|
|_ _ _ 002_S_0685
:		|_ _ _MRI.nii.gz
:		|_ _ _FDG.nii.gz
:
|_ _ _ 941_S_1311
