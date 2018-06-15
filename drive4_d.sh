CUDA_VISIBLE_DEVICES=1
spec='python driver.py --dataset davis --cross_validation 
--model tf_regression --prot_desc_path davis_data/prot_desc.csv 
--model_dir ./model_dir/model_dir4_davis_ct --cold_target 
--arithmetic_mean --aggregate toxcast --filter_threshold 1 
--intermediate_file ./interm_files/intermediate_cv_ctarget_3.csv '
eval $spec
