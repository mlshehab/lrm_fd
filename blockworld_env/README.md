## BlockWorld Experiment

### 1. Train the Policy

In order to run the experiments, we first need to train the expert policies. This can be done in the script `train_policy.py`. The arguments for the script are given below.

```bash
blockworld_env$python train_policy.py --help
usage: train_policy.py [-h] [--rm RM] [--save]

options:
  -h, --help  show this help message and exit
  --rm RM
  --save
```
The options for the rm are: `['stack', 'stack-extra', 'stack-adv']`. For example, in order to train a policy for the `(stack)` task and save it, run:
```bash
python train_policy.py --rm stack --save
```
Similarly, in order to train a policy for the `(stack-avoid)` task and save it, run:
```bash
python train_policy.py --rm stack-adv --save
```
The policy saving location is found in `config.py`, among other hyperparameters for the policy training. 

### 2. Running the main script

After training the policy, the `main.py` script takes care of simulating a dataset using the ground-truth policy, constructing the negative example set, and solving the SAT problem. The arguments are given below.
```bash
(PPO) blockworld_env$python main.py --help
usage: main.py [-h] [--depth DEPTH] [--n_traj N_TRAJ] [--rm RM] [--save]

options:
  -h, --help       show this help message and exit
  --depth DEPTH
  --n_traj N_TRAJ
  --rm RM
  --save
```
For example, in order to run the experiment for the `(stack)` task with 50K trajectories and depth $20$, run:
```bash
python main.py --rm stack --depth 20 --n_traj 50_000 --save
```