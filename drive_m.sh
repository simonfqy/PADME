CUDA_VISIBLE_DEVICES=0
spec='python driver.py --dataset metz --cross_validation 
--model graphconvreg --prot_desc_path metz_data/prot_desc.csv 
--model_dir ./model_dir/model_dir_metz_cd --cold_drug 
--arithmetic_mean --aggregate toxcast --filter_threshold 1 
--intermediate_file ./interm_files/intermediate_cv_cdrug_3.csv '
eval $spec
