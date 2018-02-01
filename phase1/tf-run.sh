#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --mem=12G
#SBATCH --time=0-00:19
#SBATCH --output=logdnn.out

module load cuda cudnn python/3.5.2
source tensorflow/bin/activate
python3 SimBoost/xgboost/DNN_est.py
