CUDA_VISIBLE_DEVICES=2
spec='python driver.py --dataset metz --cross_validation 
--model tf_regression --prot_desc_path metz_data/prot_desc.csv 
--model_dir ./model_dir/model_dir4_metz_ct --cold_target 
--arithmetic_mean --aggregate toxcast --filter_threshold 1 
--intermediate_file ./interm_files/intermediate_cv_ctarget_3.csv '
eval $spec
