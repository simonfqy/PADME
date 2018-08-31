CUDA_VISIBLE_DEVICES=3
spec='python driver.py --dataset toxcast --cross_validation 
--model tf_regression --prot_desc_path full_toxcast/prot_desc.csv 
--model_dir ./model_dir/model_dir4_tc_ct --cold_target 
--arithmetic_mean --aggregate toxcast 
--intermediate_file intermediate_cv_ctarget.csv '
eval $spec
