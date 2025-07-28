## GridWorld Experiment

### 1. Train the Policy

In order to run the experiments, we first need to train the expert policies. This can be done in the script `train_policy.py`. The arguments for the script are given below.

```bash
gridworld_env$python train_policy.py --help
usage: train_policy.py [-h] [--save]

options:
  -h, --help  show this help message and exit
  --save
```
For example, in order to train a policy and save it, run:
```bash
python train_policy.py --save
```
The policy saving location is found in `config.py`, among other hyperparameters for the policy training. 

### 2. Running the main script

After training the policy, the `main.py` script takes care of simulating a dataset using the ground-truth policy, constructing the negative example set, and solving the SAT problem. The arguments are given below.
```bash
gridworld_env$python main.py --help
usage: main.py [-h] [--depth DEPTH] [--n_traj N_TRAJ] [--umax UMAX] [--AP AP] [--use_maxsat] [--save]

options:
  -h, --help       show this help message and exit
  --depth DEPTH
  --n_traj N_TRAJ
  --umax UMAX
  --AP AP
  --use_maxsat
  --save
```
For example, to run the `(patrol)` task with umax = 3 and save the results, run:
```bash
python main.py --umax 3 --use_maxsat --save 
```


### 3. Results

| Umax | Total # of Clauses | # of Used Clauses | Total # of Rollouts | Trajectory Length | Avg. Total Reward per Rollout |
|------|--------------------|-------------------|---------------------|-------------------|-------------------------------|
|   4   |         1200           |          1200         |         10K            |           100        |             174.606                  |
|   3   |          1168          |         849          |       10K              |        100           |              151.986                 |
|   2   |         969           |            378       |          10K           |         100          |                 91.194              |
