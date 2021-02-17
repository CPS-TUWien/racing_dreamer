#!/bin/bash
name=$1
track=$2
algorithm=$3
task=$4
params=hyperparams/$algorithm.yml
steps=$5
eval_interval=20000
gpu=all

if [[ ! -z "$6" ]]
then
  gpu=$6
fi

if [[ $algorithm == 'mpo' ]] || [[ $algorithm == 'd4pg' ]]
then
    tag=acme
elif [[ $algorithm == 'sac' ]] || [[ $algorithm == 'ppo' ]]
then
    tag=sb3
else
  echo "Not a valid algorithm."
  exit 1
fi

echo "Running experiment: $name"
echo "track: $track"
echo "algorithm: $algorithm"
echo "task: $task"
echo "params: $params"
echo "steps: $steps"
echo "gpus: $gpu"

docker run --name $name --gpus $gpu -d  -v $(pwd)/logs:/app/logs --network host axel/racing:$tag python run_experiments.py --track $track --task $task --agent $algorithm --params $params --steps $steps --eval_interval 20000
