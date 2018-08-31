CUDA_VISIBLE_DEVICES=4
spec='python driver.py --dataset kiba --hyper_param_search 
--max_iter 41 --evaluate_freq 3 --patience 3 --model graphconvreg 
--early_stopping --prot_desc_path KIBA_data/prot_desc.csv 
--no_concord --no_r2 --log_file GPhypersearch_temp_kiba.log 
--model_dir ./model_dir/model_dir_sch_kiba --filter_threshold 6 
--arithmetic_mean --verbose_search --aggregate toxcast '
eval $spec
