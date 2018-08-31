CUDA_VISIBLE_DEVICES=0
spec='python driver.py --dataset kiba --cross_validation 
--model tf_regression --prot_desc_path KIBA_data/prot_desc.csv 
--model_dir ./model_dir/model_dir4_kiba_w --filter_threshold 6 
--arithmetic_mean --aggregate toxcast --split_warm 
--intermediate_file ./interm_files/intermediate_cv_warm_2.csv '
eval $spec
