CUDA_VISIBLE_DEVICES=2
spec='python3 driver.py --dataset tc_full_kinase 
--max_iter 42 --evaluate_freq 3 --patience 3 --model graphconvreg 
--prot_desc_path davis_data/prot_desc.csv --arithmetic_mean 
--prot_desc_path metz_data/prot_desc.csv --prot_desc_path KIBA_data/prot_desc.csv 
--prot_desc_path full_toxcast/prot_desc.csv --no_concord 
--model_dir ./model_dir_temp3 --plot --aggregate toxcast 
--intermediate_file intermediate_cv2.csv '
eval $spec