CUDA_VISIBLE_DEVICES=4
spec='python driver.py --dataset toxcast --model graphconvreg 
--model_dir ./model_dir/model_dir_tc_cdrug_clu_nop --no_input_protein 
--arithmetic_mean --aggregate toxcast --cross_validation --cold_drug_cluster 
--aggregate_suffix_file ./full_toxcast/Assay_UniprotID_subgroup.csv 
--intermediate_file ./interm_files/no_prot/intermediate_cv_cdrug_clu.csv '
eval $spec
