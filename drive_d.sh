CUDA_VISIBLE_DEVICES=6
spec='python driver.py --dataset davis --cross_validation 
--model graphconvreg --prot_desc_path davis_data/prot_desc.csv 
--model_dir ./model_dir/model_dir_davis_cd --cold_drug 
--arithmetic_mean --aggregate toxcast --filter_threshold 1 
--intermediate_file ./interm_files/intermediate_cv_cdrug_3.csv '
eval $spec
