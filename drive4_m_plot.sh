CUDA_VISIBLE_DEVICES=0
spec='python driver.py --dataset metz 
--model tf_regression --prot_desc_path metz_data/prot_desc.csv 
--model_dir ./model_dir/model_dir4_metz --filter_threshold 1 
--arithmetic_mean --aggregate toxcast --plot '
eval $spec
