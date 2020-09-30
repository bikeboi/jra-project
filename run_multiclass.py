import numpy as np
from experiment import ex

n_classes = [2, 4, 6, 7, 8, 9, 10]

model_suffix = "po"

for n in n_classes:
    print("N CLASS:", n)
    updates = dict(n_class=n, model_type='supervised', runs=20)
    updates_a, updates_b = (updates.copy(), updates.copy())
    updates_a.update({"A_PLUS": 0.01, "tag": "PO"})
    updates_b.update({"A_PLUS": 0.01, "A_MINUS": 0.01})

    ex.run(config_updates=updates_a, named_configs=['unmuted_config'])
    ex.run(config_updates=updates_b, named_configs=['unmuted_config'])

np.save("results/n_classes.npy", np.array(n_classes))
