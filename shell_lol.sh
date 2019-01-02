#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
##BSUB -q gputitanxpascal
### -- set the job Name --
#BSUB -J cbow_no_order_aa
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 05:00
# -- Memory --
#BSUB -R "rusage[mem=30GB]"
### -- set the email address --
#BSUB -u mahele@bioinformatics.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

# here follow the commands you want to execute

# install packages
#python3 -m pip install torchvision numpy bokeh scikit-learn torch tqdm matplotlib

nvidia-smi
# Load the cuda module
module load cuda/9.1
/appl/cuda/9.1/samples/bin/x86_64/linux/release/deviceQuery

# Load modules
module load python3/3.6.2

# Set wkdir
cd ~/Documents/language_of_life

# Activate virtual environment 
source DL-env/bin/activate

# Run script
echo "Running model..."

python scripts/CBOW_no_order_aa.py -d 'after' -pad -ws 2 -b 128 -f 'after_no_2' -e 50

echo "Done!"

# Deactivate virtual environment
deactivate
