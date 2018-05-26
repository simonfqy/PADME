CUDA_VISIBLE_DEVICES=5
spec='python3 driver.py --dataset kiba --cross_validation 
--max_iter 42 --evaluate_freq 3 --patience 3 --model tf_regression 
--prot_desc_path davis_data/prot_desc.csv --no_concord 
--prot_desc_path KIBA_data/prot_desc.csv 
--split_warm --log_file GPhypersearch_t4_kiba_w.log 
--model_dir ./model_dir4_kiba_w --filter_threshold 6 
--arithmetic_mean --aggregate toxcast '
eval $spec
