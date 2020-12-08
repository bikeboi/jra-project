# Sandbox

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("GTK3Agg")

cmap = plt.get_cmap("tab10")

n = 2
for i in range(n):
    color_ix = i/n
    print("ColorIx:", color_ix)
    plt.plot(np.random.randn(100), np.random.randn(100), '.', color=cmap(i))

plt.show()
