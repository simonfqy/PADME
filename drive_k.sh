CUDA_VISIBLE_DEVICES=4
spec='python driver.py --dataset kiba --cross_validation 
--model graphconvreg --prot_desc_path KIBA_data/prot_desc.csv 
--model_dir ./model_dir/model_dir_kiba_cd --cold_drug 
--arithmetic_mean --aggregate toxcast --filter_threshold 6 
--intermediate_file ./interm_files/intermediate_cv_cdrug.csv '
eval $spec
