import numpy as np
import os
import glob
import pandas as pd
from experiment import ex
from lib.analysis import fetch_results, compute_rates

aplus_range = np.linspace(0.001, 0.1, 20)
versions = ["muted_supervised_2", "unmuted_supervised_2"]

print(aplus_range)

def get_pred(trace):
    half = int(len(trace) / 2)
    return np.array([0 if sum(seg[:half]) > sum(seg[half:]) else 1 for seg in trace.T])


def eval_acc(trace, labels):
    pred = get_pred(trace)
    return np.mean([y == yh for y, yh in zip(labels, pred)], dtype='float')


# Clear previous results
for filename in glob.glob("results/acc*"):
    os.remove(filename)

for aplus in aplus_range:
    # Run experiments
    updates = {'A_PLUS': aplus, 'model_type': 'supervised', 'runs': 20}
    ex.run(config_updates=updates)
    ex.run(config_updates=updates, named_configs=['unmuted_config'])

    # Evaluate accuracy
    params = [np.load(f"results/{version}_params.npz") for version in versions]
    data = [fetch_results(version) for version in versions]

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
    accs.to_csv(f'results/acc_AP={aplus}.csv')

np.save("results/ap_range.npy", aplus_range)