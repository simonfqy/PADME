CUDA_VISIBLE_DEVICES=4
spec='python driver.py --dataset toxcast --cross_validation 
--model graphconvreg --cold_drug 
--prot_desc_path full_toxcast/prot_desc.csv 
--model_dir ./model_dir/model_dir_tc_cd 
--arithmetic_mean --aggregate toxcast 
--intermediate_file intermediate_cv_cdrug.csv '
eval $spec