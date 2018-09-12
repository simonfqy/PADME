CUDA_VISIBLE_DEVICES=7
spec='python driver.py --dataset toxcast --cross_validation 
--model tf_regression --prot_desc_path full_toxcast/prot_desc.csv 
--model_dir ./model_dir/model_dir4_tc_oversp_cv_ct --oversampled 
--arithmetic_mean --aggregate toxcast --cold_target 
--intermediate_file ./interm_files/intermediate_cv_ctarget_oversp.csv '
eval $spec
