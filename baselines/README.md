# Model Free Baseline Experiments

Here you can find the model-free baseline algorithms we experiment with.

## Installation

You can choose to install all the necessary dependencies locally or to build a docker
image. The latter is recommended.

### Local Installation
Install all the necessary python packages to run the experiments:

```pip install -r requirements.txt```

You can also choose to install a subset of the packages to run specific algorithms
only. For the required packages please have a look at the docker files for the
various frameworks ([Acme](./docker/Dockerfile.acme), [StableBaselines3](./docker/Dockerfile.sb3), [StableBaselines2](./docker/Dockerfile.sb2)).

### Docker
You can build all three images by running the [build script](./scripts/build_all.sh):
```bash
./scripts/build_all.sh
```
If you want to build the images one by one, you can do so by specifying the correct Dockerfile:
```shell
docker build -t tuwcps/racing:<tag> -f docker/Dockerfile.<tag>.
```

where `tag` is one of `acme`, `sb3`, or `sb2`.

## Usage

### Training

To start a training run, the main entry point is [run_experiments.py](./run_experiments.py).
You can call it with the following parameters:

```shell
python run_experiments.py \
    --track <austria|columbia|treitlstrasse_v2> \
    --task <max_progress|max_speed> \
    --agent <mpo|d4pg|sac|ppo|ppo-lstm> \
    --params <path to hyperparamfile> \ 
    --steps <steps> \
    --eval_interval <eval_interval> \
```

You can use a script ([start_experiment.sh](./scripts/start_experiment.sh)) to start a
single experiment using a docker container:

```shell
./scripts/start_experiment.sh <track> <agent> <task> <steps> <hyperparamfile>
```

For convenience, we also provide a [table](./experiments.csv) of all the model-free experiments with their
respective ID:

|id |track        |agent   |task        |
|---|-------------|--------|------------|
|1  |austria      |mpo     |max_progress|
|2  |austria      |d4pg    |max_progress|
|3  |austria      |sac     |max_progress|
|4  |austria      |ppo     |max_progress|
|5  |columbia     |mpo     |max_progress|
|6  |columbia     |d4pg    |max_progress|
|7  |columbia     |sac     |max_progress|
|8  |columbia     |ppo     |max_progress|
|9  |treitlstrasse|mpo     |max_progress|
|10 |treitlstrasse|d4pg    |max_progress|
|11 |treitlstrasse|sac     |max_progress|
|12 |treitlstrasse|ppo     |max_progress|
|13 |austria      |lstm-ppo|max_progress|
|14 |columbia     |lstm-ppo|max_progress|
|15 |treitlstrasse|lstm-ppo|max_progress|

With this table, you can launch one or more experiments using their id with this script:
[start_ids.sh](./scripts/start_ids.sh).

Example:
```bash
./scripts/start_ids 1 2 3 4 13
```
This launches training runs of all agents on austria.

### Hyperparameter Tuning

We also conducted some experiments to find good hyperparameter sets for the
model-free algorithms. For this, we used the hyperparameter optimization framework [Optuna](https://optuna.readthedocs.io/en/stable/).

For hyperparameter optimization studies, the main entry point is [run_tuning.py](./run_tuning.py).
You can call it with the following arguments:
```shell
 python run_tuning.py
            --study_name <study_name>
            --agent <mpo|d4pg|sac|ppo>
            --track <austria|columbia|treitlstrasse_v2>
            --task <max_progress|max_speed>
            --tunable_params hyperparams/tuning/<mpo|d4pg|sac|ppo>.yml
            --default_params hyperparams/default/<mpo|d4pg|sac|ppo>.yml
            --steps <steps per epoch>
            --epochs <reporting intervals>
            --trials <number of runs>
            --logdir <log-directory>
            --storage <storage-string> (see https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html)
```

To launch a study using docker containers, you can use the [start_study.sh](./scripts/start_study.sh) script:
```shell
./scripts/start_study.sh <study_name> <agent> <nr_of_parallel_tuning_instances>
```
This starts a MySQL database in a docker container and a number of workers runing the
optimization process (for details see: [study-compose.yml](./docker/study-compose.yml)).
