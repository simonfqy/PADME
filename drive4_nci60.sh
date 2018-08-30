CUDA_VISIBLE_DEVICES=2
spec='python driver.py --dataset nci60 --prot_desc_path davis_data/prot_desc.csv 
--model tf_regression --prot_desc_path full_toxcast/prot_desc.csv 
--prot_desc_path metz_data/prot_desc.csv --prot_desc_path KIBA_data/prot_desc.csv 
--model_dir ./model_dir/model_dir4_tc_w --arithmetic_mean --aggregate toxcast 
--predict_only --csv_out ./NCI60_data/preds_all_tc_ecfp.csv '
eval $spec
