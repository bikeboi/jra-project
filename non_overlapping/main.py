"""
Non-overlapping inputs test
"""

import matplotlib.pyplot as plt
import pickle
from non_overlapping.util import *
from models.stdp import gary_stdp
from brian2.only import *

# Depending on your OS this may or may not break. Change backend as needed
plt.switch_backend("GTK3Agg")

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'image.cmap': 'gray',
})

# Neuron model
NEURON = """
dv/dt = (-v + v_rest + Isyn)/tau : volt
dIsyn/dt = -Isyn/tau_syn : volt
theta = -50*mV : volt
v_rest = -55*mV : volt
tau = 50*ms : second
tau_syn = 2*ms : second
"""

INHIBITOR = """
dcount/dt = -count/(10*ms) : 1
thresh = 100 : 1
"""

start_scope()
defaultclock.dt = 0.1 * ms

# -- SIMULATION PARAMETERS --
N_CLASS = 2
SAMPLE_PERIOD = 500
N_ITER = 3
T_VAR = 10
TRAIN_RATIO = .5
N_EPOCH = 5
TRAIN_PERIOD = N_CLASS * 20 * TRAIN_RATIO * N_EPOCH * SAMPLE_PERIOD
TEST_PERIOD = N_CLASS * 20 * (1 - TRAIN_RATIO) * SAMPLE_PERIOD
DURATION = (TRAIN_PERIOD + TEST_PERIOD) * ms

# -- DATA --
images = load_images(N_CLASS)
D_IMAGE = images.shape[-1]
spikes, labels, n_train = encode_dataset(images, TRAIN_RATIO, N_EPOCH, t_var=T_VAR, n_iter=N_ITER, sample_period=SAMPLE_PERIOD)

# -- POPULATIONS --
n_pn = N_CLASS * D_IMAGE
n_kc = 15 * n_pn

# Projection Neurons (PNs)
pns = SpikeGeneratorGroup(n_pn, spikes[:, 0], spikes[:, 1] * ms)
sm_pn = SpikeMonitor(pns)

# Kenyon Cells (KCs)
kcs = NeuronGroup(n_kc, NEURON, threshold='v >= theta', reset='v = v_rest', method='exact')
kcs.v = 'v_rest'
kcs.Isyn = 0.1 * mV
sm_kc = SpikeMonitor(kcs)
vm_kc = StateMonitor(kcs, 'v', record=[0])
pm_kc = PopulationRateMonitor(kcs)

# Inhibitor (APL or Giant Gabaergic neuron?)
inh = NeuronGroup(1, INHIBITOR, threshold='count > thresh', reset='count = 0', method='exact')
inh.count = 0
vm_inh = StateMonitor(inh, 'count', True)

# Mushroom Body Output Neurons (MBONs)
mbon = NeuronGroup(N_CLASS, NEURON, threshold='v >= theta', reset='v = v_rest', method='exact')
mbon.v = 'v_rest'
sm_mbon = SpikeMonitor(mbon)
vm_mbon = StateMonitor(mbon, 'v', record=True)

# -- Connections --
# Feedforward PN->KC
pn_kc = Synapses(pns, kcs, 'g : volt', on_pre='Isyn += g', delay=2 * ms)
pn_kc.connect(p=0.1)
pn_kc.g = 50 * mV

# Feedforward KC->INH
kc_inh = Synapses(kcs, inh, on_pre='count += 1')
kc_inh.connect()

# Feedback INH->KC
inh_kc = Synapses(inh, kcs, 'g : volt', on_pre='Isyn = (Isyn/g)*mV')
inh_kc.connect()
inh_kc.g = 100 * mV

# Feedforward KC->MBON
G_MIN = 0.0
G_MAX = 1.0
kc_mbon = Synapses(kcs, mbon, gary_stdp.model(), on_pre=gary_stdp.on_pre(), on_post=gary_stdp.on_post())
kc_mbon.connect()
kc_mbon.aplus = 0.01
kc_mbon.aminus = 0.1
kc_mbon.gmin = G_MIN
kc_mbon.gmax = G_MAX
kc_mbon.tpre = kc_mbon.tpost = -np.inf*ms
kc_mbon.tplus = N_ITER/2 * 50 * ms
kc_mbon.tminus = SAMPLE_PERIOD * ms
kc_mbon.tmax = TRAIN_PERIOD*ms

print(TRAIN_PERIOD)

# Sample weights to measure
n_conns = kc_mbon.g.shape[0]
n_sample = int(n_conns * 0.25)
weight_ixs = np.random.choice(n_conns, n_sample, replace=False)
weight_sample_mbons = kc_mbon.j[weight_ixs]
sm_weight = StateMonitor(kc_mbon, 'g', record=weight_ixs, dt=SAMPLE_PERIOD*ms)

# MBON mutual inhibition
mbon_mbon = Synapses(mbon, mbon, 'g : volt', on_pre='Isyn -= g')
mbon_mbon.connect()
mbon_mbon.g = (25/N_CLASS)*mV

# -- SUPERVISION --
sup_ixs = labels.copy()[:n_train]
sup_times = (np.arange(n_train) * SAMPLE_PERIOD + N_ITER/2 * 50)*ms  # Spike in the center of all input spike times
sups = SpikeGeneratorGroup(N_CLASS, sup_ixs, sup_times)
sm_sups = SpikeMonitor(sups)

sups_mbon = Synapses(sups, mbon, 'g : volt', on_pre='v += (theta-v) +1*mV')
sups_mbon.connect(j='i')
sups_mbon.g = 10*mV

# -- SIMULATION --
print("Running Simulation")
run(DURATION, report='text', report_period=SAMPLE_PERIOD*ms)

pn_spikes = get_trains(sm_pn)
kc_spikes = get_trains(sm_kc)
mbon_spikes = get_trains(sm_mbon)
sup_spikes = get_trains(sm_sups)

# -- SAVE RESULTS --
tag = f"results/output_{N_CLASS}.pickle"
results = {
    'duration': DURATION/ms,
    'labels': labels,
    'kc': kc_spikes,
    'mbon': mbon_spikes,
    'train_ratio': TRAIN_RATIO,
    'sample_period': SAMPLE_PERIOD,
    'n_iter': N_ITER,
    'n_epoch': N_EPOCH,
}

with open(tag, 'wb') as file:
    pickle.dump(results, file)

# -- PLOTTING --
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)

# PN spikes
ax1.eventplot(pn_spikes)
ax1.set_title("Input Spikes")

# KC Spikes
ax2.eventplot(kc_spikes)
ax2.set_title("KC Spikes")

# MBON Spikes
ax3.eventplot(mbon_spikes)
ax3.set_title("Output Spikes")

# MBON voltage traces
ax4.plot(vm_mbon.t/ms, vm_mbon.v[0]/mV, label='mbon-1')
ax4.plot(vm_mbon.t/ms, vm_mbon.v[1]/mV, label='mbon-2')

# Weight Changes
cmap = plt.get_cmap("Accent")
for i in range(N_CLASS):
    color_ix = i/N_CLASS
    ax5.plot(sm_weight.t/ms, sm_weight.g[weight_sample_mbons == i].T, '.', color=cmap(color_ix), alpha=0.1)
    ax5.plot(sm_weight.t/ms, sm_weight.g[weight_sample_mbons == i].mean(axis=0), label=i, color=cmap(color_ix))

ax5.legend()

ax5.set_title(f"Weight Change $(g_\mathrm{{min}} = {G_MIN}, g_\mathrm{{max}} = {G_MAX})$")
ax5.set_ylabel("$\Delta w$")
plt.axvline(TRAIN_PERIOD, 0, 1, linestyle=':', color='k')

fig.suptitle(f"$N_\mathrm{{class}} = {N_CLASS}$")

plt.show()
