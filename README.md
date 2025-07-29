# Learning Reward Machines From Partially Observed Policies

This repository contains code for our paper: [Learning Reward Machines From Partially Observed Policies](https://arxiv.org/pdf/2502.03762), In Review,2025.

Create a virtual environment and activate it using:
```bash
cd ./lrm_fd
conda env create -f lrm.yml -n lrm
conda activate lrm
```

Each experiment from Section 5 of the paper is implemented in its own folder.

- Section 5.1 --> `./gridwolrd_env`
- Section 5.2 --> `./blockworld_env`
- Section 5.3 --> `./reacher_env`
- Section 5.4 --> `./labyrinth_env`

Generally, to run an experiment with the default hyperparameters, run:

```bash
cd ./{world}_env # world = {gridworld,blockworld,reacher,labyrinth}
python main.py
```
