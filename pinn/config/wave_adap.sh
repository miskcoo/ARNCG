#!/bin/bash

pde=wave
seeds=(0 1 2 3 4 5 6 7)
loss=mse
n_neuron=200
n_layers=4
num_x=257
num_t=101
num_res=10000
opt=adam_lbfgs_nncg_adap
switch_epoch_lbfgs=1000
switch_epoch_nncg=3000
adam_lr=0.0001
precond_update_freq=20
epochs=${4:-"5000"}
nncg_thetas=(1.0)
nncg_rank=60
beta=5
devices=(0 1 2 3 4 5 6 7)
proj=$1
max_parallel_jobs=8
dtype=${2:-"float32"}
max_time=${3:-"86400"} # 1 day

background_pids=()
current_device=0

interrupted=0  # Flag to indicate if Ctrl+C is pressed

# Function to handle SIGINT (Ctrl+C)
cleanup() {
    echo "Interrupt received, stopping background jobs..."
    interrupted=1  # Set the flag
    for pid in "${background_pids[@]}"; do
        kill $pid 2>/dev/null
    done
}

# Trap SIGINT
trap cleanup SIGINT

for nncg_theta in "${nncg_thetas[@]}"
do
    for seed in "${seeds[@]}"
    do

        if [ $interrupted -eq 0 ]; then  # Check if Ctrl+C has been pressed
            device=${devices[current_device]}
            current_device=$(( (current_device + 1) % ${#devices[@]} ))

            wandb_name="adam-lbfgs-adap_adap-theta-${nncg_theta}_seed-${seed}"
            python run_experiment.py --seed $seed --pde $pde --pde_params beta $beta --opt $opt \
                --opt_params adam_lr $adam_lr lbfgs_history_size 100 switch_epoch_lbfgs $switch_epoch_lbfgs switch_epoch_nncg $switch_epoch_nncg precond_update_freq $precond_update_freq nncg_cg_maxiter 1000 nncg_use_precond 1 nncg_rank $nncg_rank nncg_theta $nncg_theta --num_layers $n_layers --num_neurons $n_neuron \
                --loss $loss --num_x $num_x --num_t $num_t --num_res $num_res --epochs $epochs --wandb_project $proj --wandb_name $wandb_name \
                --device $device --dtype $dtype --max_time ${max_time} &

            background_pids+=($!)

            # Limit the number of parallel jobs
            while [ $(jobs | wc -l) -ge $max_parallel_jobs ]; do
                wait -n
                # Clean up finished jobs from the list
                for i in ${!background_pids[@]}; do
                    if ! kill -0 ${background_pids[$i]} 2> /dev/null; then
                        unset 'background_pids[$i]'
                    fi
                done
            done
        fi
    done
done
