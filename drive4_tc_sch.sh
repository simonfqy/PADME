CUDA_VISIBLE_DEVICES=2
spec='python driver.py --dataset toxcast --hyper_param_search 
--max_iter 1 --evaluate_freq 1 --patience 1 --model tf_regression 
--early_stopping --prot_desc_path full_toxcast/prot_desc.csv --no_r2 
--log_file GPhypersearch_t4_tc_toy.log 
--model_dir ./model_dir/model_dir4_tc_sch_toy 
--arithmetic_mean --verbose_search --aggregate toxcast '
eval $spec
