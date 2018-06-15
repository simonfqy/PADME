CUDA_VISIBLE_DEVICES=5
spec='python driver.py --dataset metz --cross_validation 
--model tf_regression --prot_desc_path metz_data/prot_desc.csv 
--split_warm --model_dir ./model_dir/model_dir4_metz_w 
--filter_threshold 1 --arithmetic_mean --aggregate toxcast 
--intermediate_file ./interm_files/intermediate_cv_warm_3.csv '
eval $spec
