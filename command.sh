#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p v100
#SBATCH --qos=v100
#SBATCH -J myFirstJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
singularity exec env_22_01 python zmj/test.py
