#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J Capture24_finetuning
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=128GB]"
### Number of hours needed
#BSUB -N 
<<<<<<< HEAD
#BSUB -W 01:00
=======
#BSUB -W 12:00
>>>>>>> 49a7722 (job script)
### added outputs and errors to files
#BSUB -o logs/Output_Capture_24_finetuning_%J.out
#BSUB -e logs/Error_Capture_24_finetuning_%J.err

module load python3/3.7.14
source ssl-env/bin/activate

<<<<<<< HEAD
python3 downstream_task_evaluation.py evaluation.num_epoch=200 data=capture24_10s data.data_root=/work3/theb/timeseries/capture24_100hz_w10_o0 evaluation=all
=======
python3 downstream_task_evaluation.py evaluation.num_epoch=200 data=capture24_10s data.data_root=/work3/theb/timeseries/capture24_100hz_w10_o0 evaluation=all num_split=5 evaluation.flip_net_path="/zhome/89/a/117273/Desktop/ssl-wearables/model_check_point/mtl_best.mdl" gpu=0 is_verbose=True evaluation.learning_rate=1e-4
>>>>>>> 49a7722 (job script)
