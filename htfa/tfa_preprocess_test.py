from tfa_preprocess import get_all_nifti,gen_intersection_mask,preprocess_run

in_path='/home/hadoop/TFA/tests/115pieman/'
out_path='/home/hadoop/TFA/tests/115pieman_labels/'
second_mask='/home/hadoop/TFA/tests/avg152T1_gray_3mm.nii'
groups=['Intact','Paragraph','Resting','Word']
threshold=100
nifti_files=get_all_nifti(in_path)
mask_file,_=gen_intersection_mask(in_path,nifti_files,second_mask,threshold)
#mask_file=in_path + 'intersection.img'
preprocess_run(in_path,out_path,nifti_files,mask_file,groups)

