# PADME: A Deep Learning-Based framework for Drug-Target Interaction Prediction 
This is the repository containing the source code for my Master's thesis research, namely predicting drug-target interaction using Deep Neural Networks. The name PADME stands for "Protein And Drug Molecule interaction prEdiction", which also happened to be the heroine of Star Wars Prequel Trilogy. The paper can be found here: https://arxiv.org/abs/1807.09741

It currently depends on a version of DeepChem Python package released in November 2017. I will need to make major modifications to it such that it would be compatible with the current version of DeepChem after I am done with my first version of the current paper. The `dcCustom` folder is a package, inheriting some classes from DeepChem. Some of the implementations are customized, so I named it dcCustom, which means "Customized version of DeepChem". 

The Python script `driver.py` at the top level is in charge of calling functions in `dcCustom` to execute the program. I assume using a Linux system, the `.sh` files call `driver.py`, each `.sh` file starts with the word `drive`, and specifies the different options that should be passed to the program. The options would include a dataset to be analyzed, model to be used, whether cross validation should be performed, etc. Like DeepChem, PADME cannot use multiple GPUs to parallelize the task, so using one GPU for one process is the most efficient choice, otherwise extra GPUs would have their memory completely occupied but not doing any useful work, only 1 GPU is the workhorse. For this purpose, `CUDA_VISIBLE_DEVICES` was specified in each `.sh` file, such that we can take advantage of multiple GPUs, each one running a specific process. To run the program, simply type the path to the corresponding shell script in the command line in Linux.

The protein descriptors used is PSC, Protein Sequence Composition descriptor, which are stored as files in the respective dataset folders, like `/full_toxcast`. You can specify the path of the protein sequence descriptor files in the `.sh` scripts. 

Currently it works fine for `graphconvreg`, `weave_regression`, `tf_regression`, and `mpnn`. I will need updates to the classification models so that it would work correctly for them too, like `weave`, `graphconv`, etc. 

You must first have `DeepChem` installed for `PADME` to work correctly, which in turn requires you to install `TensorFlow`.

Other folders like `oldCode` and `phase1` are not related to PADME, they are for the first phase of my project. You can neglect them.

# Built with
Python - Process data and constructing deep learning model

# Author
simonfqy (Qingyuan Feng)
