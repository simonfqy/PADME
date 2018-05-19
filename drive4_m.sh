CUDA_VISIBLE_DEVICES=7
spec='python3 driver.py --dataset metz --cross_validation 
--max_iter 42 --evaluate_freq 3 --patience 3 --model tf_regression 
--prot_desc_path davis_data/prot_desc.csv --no_concord 
--prot_desc_path metz_data/prot_desc.csv --prot_desc_path KIBA_data/prot_desc.csv 
--prot_desc_path full_toxcast/prot_desc.csv 
--log_file GPhypersearch_t4_metz.log --model_dir ./model_dir4_metz 
--plot --arithmetic_mean --aggregate toxcast '
eval $spec
