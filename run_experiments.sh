#!/usr/bin/env bash
set -e  # Exit on error and print each command

echo "========================================"
echo "Current working directory: $(pwd)"
echo "Setting up conda environment (this may take ~2 minutes the first time)..."
echo "========================================"

# Create and activate conda environment
if conda env list | grep -q "lrm"; then
    echo "Conda environment 'lrm' already exists. Updating it..."
    conda env update -f lrm.yml -n lrm > /dev/null 2>&1
else
    echo "Creating new conda environment 'lrm'..."
    conda env create -f lrm.yml -n lrm > /dev/null 2>&1
fi

# Initialize conda for the current shell
eval "$(conda shell.bash hook)"
conda activate lrm


rm -rf results/*


########################################
# Table 1: Patrol Tasks
########################################
echo ""
echo "========================================"
echo "Reproducing Table 1: Patrol Tasks"
echo "========================================"

cd old_experiments

echo "PATROL WITH NO SUTTERING ... \n"  >> ../results/TABLE_1.txt
python -u patrol.py --depth 6 --print_solutions --non_stuttering >> ../results/TABLE_1.txt
echo "" >> ../results/TABLE_1.txt
echo "" >> ../results/TABLE_1.txt
echo "PATROL WITH SUTTERING ... \n"  >> ../results/TABLE_1.txt
python -u patrol.py --depth 6 --print_solutions >> ../results/TABLE_1.txt
echo "" >> ../results/TABLE_1.txt
echo "" >> ../results/TABLE_1.txt

echo "PATROL WITH HALLWAY ... \n"  >> ../results/TABLE_1.txt
# python -u patrol_hallway.py --depth 9 >> ../../results/TABLE_1.txt

########################################
# Table 2: Stacking Tasks
########################################
echo ""
echo "========================================"
echo "Reproducing Table 2: Stacking Tasks"
echo "========================================"
echo "StACK ... \n" >> ../results/TABLE_2.txt
python -u stack.py --depth 10 >> ../results/TABLE_2.txt
echo "" >> ../results/TABLE_2.txt
echo "" >> ../results/TABLE_2.txt
echo "StACK AVOID ... \n" >> ../results/TABLE_2.txt
python -u stack_avoid.py --depth 8 >> ../results/TABLE_2.txt

########################################
# Table 3 and Figure 5.c: Reacher Env
########################################
cd ../reacher_env

echo ""
echo "========================================"
echo "Reproducing Table 3: Reacher Environment"
echo "========================================"

python -u main.py --alpha 0.001 0.0001 0.00001 >> ../results/TABLE_3.txt

echo ""
echo "Computing reward for Figure 5.c ..."
python compute_rewards.py --umax 2 >> ../results/fig5c_rewards.txt 

########################################
# Tables 4 and 5: Blockworld Environment
########################################
cd ../blockworld_env

echo ""
echo "========================================"
echo "Reproducing Table 4: Stack Reward Machine"
echo "========================================"
python -u main.py --rm stack --n_traj 1000 3000 5000 10000 100_000 1_000_000 >> ../results/TABLE_4.txt 

echo ""
echo "========================================"
echo "Reproducing Table 5: Stack-Adv Reward Machine"
echo "========================================"
python -u main.py --rm stack-adv --n_traj 100 200 500 1000 100_000 1_000_000 >> ../results/TABLE_5.txt

echo ""
echo "========================================"
echo "Reproducing Section C: Stack-Extra Reward Machine"
echo "========================================"
python -u main.py --rm stack-extra --depth 14 --n_traj 50_000 >> ../results/SECTION_C.txt

########################################
# Figure 11 and Table 6: Gridworld Environment
########################################
cd ../gridworld_env

echo ""
echo "========================================"
echo "Reproducing Figure 11: RM Learning in Gridworld"
echo "========================================"

for umax in 4 3 2; do
    echo ""
    echo "Running RM learning with umax = $umax ..."
    python -u main.py --run_rmm_learning --umax "$umax" --print_solutions >> ../results/figure11.txt
done

echo ""
echo "========================================"
echo "Reproducing Table 6: Gridworld Baselines"
echo "========================================"

python -u main.py --umax 4 >>  ../results/TABLE_6.txt    # Row 1
python -u main.py --umax 3 >>  ../results/TABLE_6.txt     # Row 2
python -u main.py --umax 2 >>  ../results/TABLE_6.txt     # Row 3
python -u main.py --use_irl >>  ../results/TABLE_6.txt     # Row 4
