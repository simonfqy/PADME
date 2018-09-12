CUDA_VISIBLE_DEVICES=6
spec='python driver.py --dataset toxcast --cross_validation 
--model tf_regression --prot_desc_path full_toxcast/prot_desc.csv 
--model_dir ./model_dir/model_dir4_tc_oversp_cv_cd --oversampled 
--arithmetic_mean --aggregate toxcast --cold_drug 
--intermediate_file ./interm_files/intermediate_cv_cdrug_oversp.csv '
eval $spec
