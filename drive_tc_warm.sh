CUDA_VISIBLE_DEVICES=2
spec='python driver.py --dataset toxcast --cross_validation 
--model graphconvreg 
--no_r2 --prot_desc_path full_toxcast/prot_desc.csv 
--log_file GPhypersearch_t4_tc_w.log 
--model_dir ./model_dir/model_dir_tc_w 
--arithmetic_mean --aggregate toxcast 
--intermediate_file intermediate_cv_warm.csv '
eval $spec