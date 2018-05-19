CUDA_VISIBLE_DEVICES=2
spec='python driver.py --dataset kiba --hyper_param_search 
--max_iter 38 --evaluate_freq 3 --patience 3 --model tf_regression 
--early_stopping --prot_desc_path davis_data/prot_desc.csv --no_concord 
--prot_desc_path metz_data/prot_desc.csv --prot_desc_path KIBA_data/prot_desc.csv 
--prot_desc_path full_toxcast/prot_desc.csv --no_r2 
--log_file GPhypersearch_t4_kiba.log --model_dir ./model_dir4_kiba_sch 
--arithmetic_mean --verbose_search --aggregate toxcast '
eval $spec
