import wandb
import json
import pandas as pd
import numpy as np
import pickle
import os

# Initialize the API
api = wandb.Api()


def collect(project_name, entity_name='<PLACEHOLDER, CHANGE IT>', cache_dir='cache', use_cache=True):

    print(f"Collecting data for project: {project_name}")

    if cache_dir and use_cache:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if os.path.exists(os.path.join(cache_dir, f"{project_name}.pkl")):
            with open(os.path.join(cache_dir, f"{project_name}.pkl"), 'rb') as f:
                data = pickle.load(f)
            return data

    runs = api.runs(f"{entity_name}/{project_name}")

    data = [] 
    for run in runs:
        # Collect the data
        data.append({
            'name': run.name,
            'config': run.config,
            'history': run.history(),
            'state': run.state,
        })

    # save the data to a file 
    if cache_dir:
        with open(os.path.join(cache_dir, f"{project_name}.pkl"), 'wb') as f:
            pickle.dump(data, f)
    return data

def summary(project_name, ignore_failed=True):
    data = collect(project_name)
    runtimes = []
    losses = []
    l2res = []
    for run in data:
        history = run['history']

        # check if the run is failed
        if ignore_failed and history['loss'].iloc[-1] == 'NaN':
            print(f"Run {run['name']} failed, skipping...")
            continue

        last_idx = history[history['loss'].ne('NaN')].index[-1]

        runtime = history['_runtime'].iloc[last_idx]
        loss = history['loss'].iloc[last_idx]
        l2re = history['test/l2re'].iloc[last_idx]

        runtimes.append(runtime)
        losses.append(loss)
        l2res.append(l2re)

    print('  runtime = ', np.mean(runtimes), '±', np.std(runtimes) / np.sqrt(len(runtimes)))
    print('  loss (mean) = ', np.mean(losses), '±', np.std(losses) / np.sqrt(len(losses)))
    print('  loss (best) = ', np.min(losses))
    print('  l2re (mean) = ', np.mean(l2res), '±', np.std(l2res) / np.sqrt(len(l2res)))
    print('  l2re (best) = ', np.min(l2res))

