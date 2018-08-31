CUDA_VISIBLE_DEVICES=0
spec='python driver.py --dataset davis 
--model tf_regression --prot_desc_path davis_data/prot_desc.csv 
--model_dir ./model_dir/model_dir4_davis --filter_threshold 1 
--arithmetic_mean --aggregate toxcast --plot '
eval $spec
