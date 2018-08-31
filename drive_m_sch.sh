CUDA_VISIBLE_DEVICES=1
spec='python driver.py --dataset metz --hyper_param_search 
--max_iter 42 --evaluate_freq 3 --patience 3 --model graphconvreg 
--early_stopping --prot_desc_path metz_data/prot_desc.csv 
--no_concord --no_r2 --log_file GPhypersearch_temp_metz_updated.log 
--model_dir ./model_dir/model_dir_sch_metz 
--arithmetic_mean --verbose_search --aggregate toxcast '
eval $spec
