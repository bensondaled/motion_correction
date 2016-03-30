#!/bin/env bash
#
#SBATCH -p all                  # partition (queue)
#SBATCH -n 5                    # number of cores
#SBATCH -t 1000                 # time (minutes)
#SBATCH -o logs/%N.%j.out       # STDOUT
#SBATCH -e logs/%N.%j.err       # STDERR
#SBATCH --mem=100000            #in MB
#SBATCH --mail-type=END,FAIL    # notifications for job done & fail

#SBATCH --mail-user=MY_PRINCETON_USERNAME@princeton.edu # ADD YOUR EMAIL ADDRESS HERE TO GET EMAIL NOTIFICATIONS

module load anacondapy/2.7
. activate motcor_env

xvfb-run --auto-servernum --server-num=1 python main.py $SLURM_ARRAY_TASK_ID
