#!/bin/bash
#SBATCH --container=el9hw
#SBATCH --partition=cm3atou_el9       
#SBATCH --job-name=Al_400k_3
#SBATCH --output=cpu_ni_%j.out
#SBATCH --error=cpu_ni_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mail-user=zeyu.dong-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/dong1/Al_400k_3

echo "Job running on compute node: $(hostname)"
echo "Current working directory: $(pwd)"

module purge
module load GCCcore/13.3.0
module load Python-bundle-PyPI/2024.06-GCCcore-13.3.0
module load CMake/3.29.3-GCCcore-13.3.0
module load FFTW/3.3.10-GCC-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load matplotlib/3.9.2

export PYTHONPATH="/home/dong1/OpenDiS/core/exadis/python:$PYTHONPATH"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=16         

echo "=========================================================="
echo "Starting High-Throughput Manager (27 Permutations for Ni)"
echo "=========================================================="

python3 -u project_manager.py