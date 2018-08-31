CUDA_VISIBLE_DEVICES=3
spec='python driver.py --dataset toxcast --hyper_param_search 
--max_iter 1 --evaluate_freq 1 --patience 1 --model graphconvreg 
--early_stopping --prot_desc_path full_toxcast/prot_desc.csv --no_r2 
--log_file GPhypersearch_temp_tc_toy.log --model_dir ./model_dir/model_dir_sch_tc_toy 
--arithmetic_mean --verbose_search --aggregate toxcast '
eval $spec
