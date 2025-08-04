### Reacher Environment
In order to run this experiment, we provide the simulated dataset [here](https://drive.google.com/file/d/1MaKUK2PcPL567rIf1LyWpeyE8Y-PF5US/view?usp=drive_link). Please download it (~1Gb) and run the following.

```bash
cd ./reacher_env
mkdir objects
cd objects
# PLACE THE FILE data_1M.pkl inside this folder
```
The options for main are below.
```bash
reacher_env$python main.py --help
usage: main.py [-h] [--alpha ALPHA] [--print]

options:
  -h, --help     show this help message and exit
  --alpha ALPHA
  --print
```
Use `--print` if you would like to see the learned reward machines. These are given using the Boolean matrices associated with each atomic proposition. 

## Reproducability
To reproduce Table 3, please run:
```bash
python main.py --alpha 0.001 0.0001 0.00001
```