CUDA_VISIBLE_DEVICES=5
spec='python driver.py --dataset davis --hyper_param_search 
--max_iter 41 --evaluate_freq 3 --patience 3 --model graphconvreg 
--early_stopping --prot_desc_path davis_data/prot_desc.csv 
--no_concord --no_r2 --log_file GPhypersearch_temp_davis_updated.log 
--model_dir ./model_dir/model_dir_sch_davis 
--arithmetic_mean --verbose_search --aggregate toxcast '
eval $spec
