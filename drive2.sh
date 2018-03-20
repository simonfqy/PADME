CUDA_VISIBLE_DEVICES=2
spec='python3 driver.py --dataset all_kinase --hyper_param_search 
--max_iter 3 --evaluate_freq 1 --patience 0
--early_stopping --prot_desc_path davis_data/prot_desc.csv 
--prot_desc_path metz_data/prot_desc.csv --prot_desc_path KIBA_data/prot_desc.csv 
--log_file GPhypersearch_t2.log --model_dir ./model_dir2'
eval $spec
