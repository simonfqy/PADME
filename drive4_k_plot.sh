CUDA_VISIBLE_DEVICES=0
spec='python driver.py --dataset kiba 
--model tf_regression --prot_desc_path KIBA_data/prot_desc.csv 
--model_dir ./model_dir/model_dir4_kiba --filter_threshold 6 
--arithmetic_mean --aggregate toxcast --plot '
eval $spec
