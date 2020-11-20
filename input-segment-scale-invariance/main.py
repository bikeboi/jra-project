"""
Sandbox to test out random ideas
"""

import imageio
import pickle
import glob
import matplotlib.pyplot as plt
from functools import reduce
import brian2.numpy_ as np
from brian2.only import *

plt.switch_backend("GTK3Agg")

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'image.cmap': 'gray',
})

# Neuron and Synapse models
NEURON = """
dv/dt = (-v + v_rest + Isyn)/tau : volt
dIsyn/dt = -Isyn/tau_syn : volt
v_rest : volt (shared)
theta : volt (shared)
tau : second (shared)
tau_syn : second (shared)
"""

SYNAPSE = "w : 1"
EXC = "Isyn += w*mV"
INH = "Isyn -= w*mV"


# Setup
start_scope()
# Population parameters
n_pn = 100
n_kc = 1000
n_iter = 10
n_epoch = 1
n_class = 5
n_trial = 5

p_fire = 1/n_class
p_conn = 20/n_pn
omega = 200

# Input dataset parameters
min_rate = 5
max_rate = 30


# Helper functions
def input_set(i_offset=0, t_offset=0 * ms):
    mk_indices = lambda i: np.tile(np.arange(n_pn // n_class), n_iter) + i*(n_pn//n_class)
    times = (np.full((n_pn // n_class, 1), 50) * np.arange(1, n_iter + 1)).flatten('F')

    indices = np.concatenate([mk_indices(i) for i in range(n_class)])
    times = np.concatenate([times*ms + i*second + t_offset for i in range(n_class)])

    return indices, times * second


def get_trains(sm: SpikeMonitor):
    return [train / ms for train in sm.spike_trains().values()]


# Projection Neurons
pn = SpikeGeneratorGroup(n_pn, *input_set(0, 0), period=n_class*second)
sm_pn = SpikeMonitor(pn)

# Kenyon Cells
kc = NeuronGroup(n_kc, NEURON, method='exact', threshold='v > theta', reset='v = v_rest', name='KC')
kc.v_rest = -55 * mV
kc.theta = -50 * mV
kc.tau = 100 * ms
kc.tau_syn = 2 * ms
kc.v = 'v_rest'
sm_kc = SpikeMonitor(kc)
vm_kc = StateMonitor(kc[0], 'v', True)
prm = PopulationRateMonitor(kc)

# GABA-ergic neuron
gn = NeuronGroup(1, NEURON, method='exact', threshold='v > theta', reset='v = v_rest', name='GN')
gn.v_rest = -55 * mV
gn.theta = -50 * mV
gn.tau = 20 * ms
gn.tau_syn = 2 * ms
gn.v = 'v_rest'
sm_gn = SpikeMonitor(gn)

# Feedback inhibition
kc_gn = Synapses(kc, gn, SYNAPSE, on_pre=EXC)
kc_gn.connect()
kc_gn.w = 10

gn_kc = Synapses(gn, kc, SYNAPSE, on_pre=INH)
gn_kc.connect()
gn_kc.w = 15

duration = n_class * n_epoch * second
results = {
    'pn': [],
    'kc': [],
    'duration': duration/ms,
    'n_class': n_class,
    'n_epoch': n_epoch,
    'n_iter': n_iter,
}

net = Network(collect())
net.store('step-1')

for trial in range(n_trial):
    print("Trial:", trial+1)
    # PN-KC Connectivity
    pn_kc = Synapses(pn, kc, SYNAPSE, on_pre=EXC)
    pn_kc.connect(p=20 / n_pn)
    pn_kc.w = omega / (2*n_pn*p_fire*p_conn)

    # Run the simulation
    net.add(pn_kc)
    net.run(duration, report='text')

    results['pn'].append(get_trains(sm_pn))
    results['kc'].append(get_trains(sm_kc))

    net.remove(pn_kc)
    net.restore('step-1')


# Save results
pickle.dump(results, open(f'spikes_{n_class}.p', 'wb'))
print("Done")

# Plot example
fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
ax1.eventplot(results['pn'][0])
ax2.eventplot(results['kc'][0])
ax1.set_ylabel("neuron index")
ax2.set_ylabel("neuron index")
ax2.set_xlabel("time [ms]")
plt.show()

