import numpy as np
import os
import glob
import pandas as pd
from experiment import ex
from lib.analysis import fetch_results, compute_rates

step = 20
val_range = np.linspace(0.005, 0.05, step)

aplus, aminus = np.meshgrid(val_range, val_range)
aplus = np.expand_dims(aplus, -1)
aminus = np.expand_dims(aminus, -1)

stdp_params = np.concatenate([aplus, aminus], axis=-1).reshape(-1, 2)

n_class = 2
versions = [f"stdp_param_sweep_supervised_{n_class}"]

def get_pred(trace):
    half = int(len(trace) / 2)
    return np.array([0 if sum(seg[:half]) > sum(seg[half:]) else 1 for seg in trace.T])


def eval_acc(trace, labels):
    pred = get_pred(trace)
    return np.mean([y == yh for y, yh in zip(labels, pred)], dtype='float')


# Clear previous results
for filename in glob.glob("results/acc_AP=*_AM=*"):
    os.remove(filename)

for version in versions:
    for filename in glob.glob(f"results/{version}/**/*"):
        os.remove(f"results/{filename}")

for aplus,aminus in stdp_params:
    # Run experiments
    updates = {'A_MINUS': aminus, 'A_PLUS': aplus,
               'model_type': 'supervised',
               'runs': 10,
               'tag': 'stdp_param_sweep'
               }
    ex.run(config_updates=updates, named_configs=['unmuted_config'])

    # Evaluate accuracy
    params = [np.load(f"results/{version}_params.npz") for version in versions]
    data = [fetch_results(version, param) for param, version in zip(params, versions)]

    rates = np.array([compute_rates(runs, params[i]) for i, runs in enumerate(data)])

    accs = pd.DataFrame()
    for i, ver in enumerate(versions):
        runs = rates[i]
        labelset = params[i]['labels']
        for run, labels in zip(runs, labelset):
            acc = eval_acc(run, labels)
            accs = accs.append(pd.DataFrame({
                "version": [ver], "accuracy": acc
            }))

    print("Saving results")
    accs.to_csv(f'results/acc_AP={aplus}_AM={aminus}.csv')

np.save("results/ap_am_range.npy", stdp_params)
