# external libraries and packages
import wandb
import argparse
import sys
import traceback
import torch

from src.train_utils import set_random_seed, train
from src.models import PINN

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='initial seed')
    parser.add_argument('--pde', type=str,
                        default='convection', help='PDE type')
    parser.add_argument('--pde_params', nargs='+', type=str,
                        default=None, help='PDE coefficients')
    parser.add_argument('--opt', type=str, default='lbfgs',
                        help='optimizer to use')
    parser.add_argument('--opt_params', nargs='+', type=str,
                        default=None, help='optimizer parameters')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of layers of the neural net')
    parser.add_argument('--num_neurons', type=int, default=50,
                        help='number of neurons per layer')
    parser.add_argument('--loss', type=str, default='mse',
                        help='type of loss function')
    parser.add_argument('--num_x', type=int, default=257,
                        help='number of spatial sample points (power of 2 + 1)')
    parser.add_argument('--num_t', type=int, default=101,
                        help='number of temporal sample points')
    parser.add_argument('--max_time', type=int, default=3600 * 48,  # 2 days
                        help='maximum time to run the experiment')
    parser.add_argument('--num_res', type=int, default=10000,
                        help='number of sampled residual points')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to run')
    parser.add_argument('--wandb_project', type=str,
                        default='pinns', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str,
                        default='miskcoo-tsinghua-university', help='W&B entity name')
    parser.add_argument('--wandb_name', type=str,
                        default=None, help='W&B name')
    parser.add_argument('--device', type=str, default=0, help='GPU to use')
    parser.add_argument('--dtype', type=str, default="float32", help='dtype to use')

    # Extract arguments from parser
    args = parser.parse_args()
    # set initial seed
    initial_seed = args.seed
    set_random_seed(initial_seed)

    # organize arguments for the experiment into a dictionary for logging purpose
    experiment_args = {
        "initial_seed": args.seed,
        "pde": args.pde,
        "pde_params": args.pde_params,
        "opt": args.opt,
        "opt_params": args.opt_params,
        "num_layers": args.num_layers,
        "num_neurons": args.num_neurons,
        "loss": args.loss,
        "num_x": args.num_x,
        "num_t": args.num_t,
        "num_res": args.num_res, 
        "max_time": args.max_time, 
        "epochs": args.epochs,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_name": args.wandb_name,
        "device": f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu',
        "dtype": torch.float64 if args.dtype == "float64" else torch.float32,
    }

    # print out arguments
    print("Seed set to: {}".format(initial_seed))
    print("Selected PDE type: {}".format(experiment_args["pde"]))
    print("Specified PDE coefficients: {}".format(
        experiment_args["pde_params"]))
    print("Optimizer to use: {}".format(experiment_args["opt"]))
    print("Specified optimizer parameters: {}".format(
        experiment_args["opt_params"]))
    print("Number of layers: {}".format(experiment_args["num_layers"]))
    print("Number of neurons per layer: {}".format(experiment_args["num_neurons"]))
    print("Number of spatial points (x): {}".format(experiment_args["num_x"]))
    print("Number of temporal points (t): {}".format(experiment_args["num_t"]))
    print("Number of random residual points to sample: {}".format(experiment_args["num_res"]))
    print("Number of epochs: {}".format(experiment_args["epochs"]))
    print("Weights and Biases project: {}".format(
        experiment_args["wandb_project"]))
    print("GPU to use: {}".format(experiment_args["device"]))
    print("dtype: {}".format(experiment_args["dtype"]))

    with wandb.init(project=experiment_args["wandb_project"], entity=experiment_args["wandb_entity"], name=experiment_args["wandb_name"], config=experiment_args):
        # initialize model
        model = PINN(in_dim=2, hidden_dim=experiment_args["num_neurons"], out_dim=1,
                     num_layer=experiment_args["num_layers"],
                     dtype=experiment_args["dtype"],
                     ).to(experiment_args["device"])
        # train the model
        try:
            train(model,
                  proj_name=experiment_args["wandb_project"],
                  pde_name=experiment_args["pde"],
                  pde_params=experiment_args["pde_params"],
                  loss_name=experiment_args["loss"],
                  opt_name=experiment_args["opt"],
                  opt_params_list=experiment_args["opt_params"],
                  n_x=experiment_args["num_x"],
                  n_t=experiment_args["num_t"],
                  n_res=experiment_args["num_res"],
                  num_epochs=experiment_args["epochs"],
                  max_time=experiment_args["max_time"],
                  device=experiment_args["device"],
                  dtype=experiment_args["dtype"],
                )
        # log error and traceback info to W&B, and exit gracefully
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise e

if __name__ == "__main__":
    main()
