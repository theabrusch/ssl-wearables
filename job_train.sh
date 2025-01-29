#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_smallmodel_pt
### number of core
#BSUB -n 10 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=8GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 02:00
### added outputs and errors to files
#BSUB -o logs/Output_Capture_24_finetuning_%J.out
#BSUB -e logs/Error_Capture_24_finetuning_%J.err

module load python3/3.7.14
source ssl-env/bin/activate

python3 downstream_task_evaluation.py evaluation.num_epoch=200 data=capture24_10s data.data_root=/work3/theb/timeseries/capture24_30hz_full data.batch_size=1000 evaluation=all output_path=/work3/theb/outputs/ssl-wearables_diffmod/ num_split=5 train_model=true evaluate_all_data=false evaluation.flip_net_path="/zhome/89/a/117273/Desktop/ssl-wearables/model_check_point/aFalse_pTrue_tTrue.mdl" gpu=0 is_verbose=True evaluation.learning_rate=1e-4 test_mode=False data.output_size=4 prefix='pre' evaluation.freeze_weight=true mixup=false save_outputs=true
