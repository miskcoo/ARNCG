#!/bin/bash

pde=reaction
seeds=(0 1 2 3 4 5 6 7)
loss=mse
n_neuron=200
n_layers=4
num_x=257
num_t=101
num_res=10000
opt=adam_lbfgs_nncg
switch_epoch_lbfgs=1000
switch_epoch_nncg=3000 # args in the original repo
# switch_epoch_nncg=41000 # args in the paper and figures
adam_lr=0.0001
# adam_lr=(0.00001 0.0001 0.001 0.01 0.1) # they do grid search for adam in the paper
precond_update_freq=20
nncg_mus=($2)
nncg_rank=60
epochs=${5:-"5000"}
# epochs=50000 # args in the paper and figures
rho=5
devices=(0 1 2 3 4 5 6 7)
proj=$1
max_parallel_jobs=8
dtype=${3:-"float32"}
max_time=${4:-"86400"} # 1 day

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

for nncg_mu in "${nncg_mus[@]}"
do
    for seed in "${seeds[@]}"
    do

        if [ $interrupted -eq 0 ]; then  # Check if Ctrl+C has been pressed
            device=${devices[current_device]}
            current_device=$(( (current_device + 1) % ${#devices[@]} ))

            wandb_name="adam-lbfgs-nncg_nncg-mu-${nncg_mu}_seed-${seed}"
            python run_experiment.py --seed $seed --pde $pde --pde_params rho $rho --opt $opt \
                --opt_params adam_lr $adam_lr lbfgs_history_size 100 switch_epoch_lbfgs $switch_epoch_lbfgs switch_epoch_nncg $switch_epoch_nncg precond_update_freq $precond_update_freq nncg_rank $nncg_rank nncg_mu $nncg_mu --num_layers $n_layers --num_neurons $n_neuron \
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
