CUDA_VISIBLE_DEVICES=0
spec='python driver.py --dataset kiba --cross_validation 
--model graphconvreg --prot_desc_path KIBA_data/prot_desc.csv 
--split_warm --model_dir ./model_dir/model_dir_kiba_w 
--filter_threshold 6 --arithmetic_mean --aggregate toxcast 
--intermediate_file ./interm_files/intermediate_cv_warm.csv '
eval $spec
