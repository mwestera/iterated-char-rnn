#!/bin/bash
#SBATCH --job-name="cluster1"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-15:00
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mem 4000
mkdir -p logs
module load CUDA/8.0.61
module load Python/3.6.4-foss-2017a
module load PyTorch/0.4.0-foss-2017a-Python-3.6.4
module load scikit-learn/0.18.1-foss-2017a-Python-3.6.4
module load numpy/1.14.0-foss-2017a-Python-3.6.4
python generate_script.py 100 config.ini > cluster1.sh
cd $PWD
. cluster1.sh