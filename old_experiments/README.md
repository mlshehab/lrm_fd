### Old Experiments

This folder contains the scripts that reproduce Tables 1 and 2 of the paper. The code here is from a previous version of the repository and is not consistent with the structures intorudced in the rest of the repository. These scripts are thus put in their own folder here and there is no plan to reintegrate in the mean time. 

> **Warning**: These scripts generally take a long time (>5 hrs for some) as they require exhausting a prefix tree policy.

To reproduce Table 1, please run:

```bash
python patrol.py --depth 6 --print_solutions --non_stuttering # ROW 1 of TABLE 1  
python patrol.py --depth 6 --print_solutions                  # ROW 2 of TABLE 1
```

To reproduce Table 2, please run:
```bash

```