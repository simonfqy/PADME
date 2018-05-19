CUDA_VISIBLE_DEVICES=3
spec='python3 driver.py --dataset tc_kinase --hyper_param_search 
--max_iter 42 --evaluate_freq 3 --patience 3 --no_concord 
--early_stopping --prot_desc_path davis_data/prot_desc.csv 
--prot_desc_path metz_data/prot_desc.csv --prot_desc_path KIBA_data/prot_desc.csv 
--prot_desc_path toxcast_data/binding_prot_desc.csv 
--log_file GPhypersearch_t2.log --model_dir ./model_dir2 
--cross_validation '
eval $spec
