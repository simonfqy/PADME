CUDA_VISIBLE_DEVICES=2
spec='python driver.py --dataset toxcast 
--model tf_regression --prot_desc_path full_toxcast/prot_desc.csv 
--model_dir ./model_dir/model_dir4_tc_w 
--arithmetic_mean --aggregate toxcast 
--intermediate_file intermediate_cv_warm.csv '
eval $spec
