#!/bin/bash
#SBATCH --job-name=webat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu2
#SBATCH --output=log/out_%j.out  # %j will be replaced with the job ID

## Execute the python scripts
module load conda
source /opt/anaconda/4.10.1/etc/profile.d/conda.sh
conda activate tf_gpu02
cd ../

# Run tracking script
python3.9 test_object_tracker.py ../videos/2023-07-07_09_00_00_055/Far_Left\ 2023-07-06_18_59_59_820.asf
python3.9 test_object_tracker.py ../videos/2023-07-07_09_00_00_055/Middle\ 2023-07-06_18_59_59_952.asf
python3.9 test_object_tracker.py ../videos/2023-07-07_09_00_00_055/Far_Right\ 2023-07-06_19_00_00_259.asf

# Run depth analysis - either from three cameras or two cameras
python3.9 test_depth_analysis.py --left ../results/Far_Left\ 2023-07-06_18_59_59_820/Far_Left\ 2023-07-06_18_59_59_820_bat.csv --middle ../results/Middle\ 2023-07-06_18_59_59_952/Middle\ 2023-07-06_18_59_59_952_bat.csv --right ../results/Far_Right\ 2023-07-06_19_00_00_259/Far_Right\ 2023-07-06_19_00_00_259_bat.csv
python3.9 test_depth_analysis_with_two.py --middle ../results/Middle\ 2023-07-06_18_59_59_952/Middle\ 2023-07-06_18_59_59_952_bat.csv --right ../results/Far_Right\ 2023-07-06_19_00_00_259/Far_Right\ 2023-07-06_19_00_00_259_bat.csv

# 3D visualization
python3.9 test_visualize_3d.py ../final_training_set/Full_bat_Middle\ 2023-07-06_18_59_59_952_bat_Far_Right\ 2023-07-06_19_00_00_259_bat.xlsx