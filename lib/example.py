import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pynn_genn as sim

from cells import IF_curr_exp_adapt
from analysis import calculate_activity

matplotlib.use("GTK3Agg")

steps = 1000
delta_t = 1.0


decay_rates = np.linspace(120, 1000, 50)
firing_rates = []

for i, rate in enumerate(decay_rates):
    sim.setup(delta_t)

    pop = sim.Population(1, IF_curr_exp_adapt(tau_threshold=10))
    pop.record(['v', 'v_thresh_adapt', 'spikes'])
    current = sim.StepCurrentSource(
        times=[0, steps-100], amplitudes=[10.0, 0.0])
    current.inject_into(pop)

    pop.set(tau_threshold=rate)

    sim.run(steps)

    spikes = pop.get_data('spikes', clear=True).segments[0].spiketrains

    interval_step = 50
    intervals = np.arange(0, steps, interval_step)
    activity = calculate_activity(spikes, intervals, interval_step)[0] / 50

    firing_rate = np.mean(activity)
    firing_rates.append(firing_rate)

    sim.end()

plt.plot(decay_rates, firing_rates)
plt.title("Firing rate by threshold decay")
plt.xlabel("decay rate (ms)")
plt.ylabel("firing rate (sp/50ms)")

plt.show()

sim.end()
