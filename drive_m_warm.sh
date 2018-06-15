CUDA_VISIBLE_DEVICES=3
spec='python driver.py --dataset metz --cross_validation 
--model graphconvreg --prot_desc_path metz_data/prot_desc.csv 
--model_dir ./model_dir/model_dir_metz_w --filter_threshold 1 
--arithmetic_mean --aggregate toxcast --split_warm 
--intermediate_file ./interm_files/intermediate_cv_warm_3.csv '
eval $spec
