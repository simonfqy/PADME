CUDA_VISIBLE_DEVICES=4
spec='python3 driver.py --dataset metz --cross_validation 
--max_iter 42 --evaluate_freq 3 --patience 3 --model tf_regression 
--prot_desc_path davis_data/prot_desc.csv --no_concord 
--prot_desc_path metz_data/prot_desc.csv 
--split_warm --log_file GPhypersearch_t4_metz_w.log 
--model_dir ./model_dir4_metz_w --filter_threshold 1 
--arithmetic_mean --aggregate toxcast '
eval $spec
