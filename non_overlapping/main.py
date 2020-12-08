"""
Non-overlapping inputs test
"""

import matplotlib.pyplot as plt
import pickle
from non_overlapping.util import *
from models.stdp import gary_stdp
from brian2.only import *

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
TRAIN_RATIO = .7
N_EPOCH = 2
DURATION = (N_CLASS * 20 * TRAIN_RATIO * N_EPOCH + N_CLASS * 20 * (1-TRAIN_RATIO)) * SAMPLE_PERIOD * ms

# -- DATA --
images = load_images(N_CLASS)
D_IMAGE = images.shape[-1]
spikes, labels = encode_dataset(images, TRAIN_RATIO, N_EPOCH, t_var=T_VAR, n_iter=N_ITER, sample_period=SAMPLE_PERIOD)

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
kc_mbon = Synapses(kcs, mbon, gary_stdp.model(), on_pre=gary_stdp.on_pre(), on_post=gary_stdp.on_post())
kc_mbon.connect()
kc_mbon.aplus = 0.01
kc_mbon.aminus = 0.075
kc_mbon.tpre = kc_mbon.tpost = -np.inf*ms
kc_mbon.gmin = 0
kc_mbon.gmax = 0.1
kc_mbon.tplus = N_ITER/2 * 50 * ms
kc_mbon.tminus = SAMPLE_PERIOD*ms

# MBON mutual inhibition
mbon_mbon = Synapses(mbon, mbon, 'g : volt', on_pre='Isyn -= g')
mbon_mbon.connect()
mbon_mbon.g = 1*mV

# -- SUPERVISION --
n_train = int(len(labels) * TRAIN_RATIO)  # Number of training samples
sup_ixs = labels.copy()[:n_train]
sup_times = (np.arange(n_train) * 500 + N_ITER/2 * 50)*ms  # Spike in the center of all input spike times
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
}

with open(tag, 'wb') as file:
    pickle.dump(results, file)

# -- PLOTTING --
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)

ax1.eventplot(pn_spikes)
ax1.set_title("Input Spikes")
ax2.eventplot(kc_spikes)
ax2.set_title("KC Spikes")
ax3.eventplot(mbon_spikes)
ax3.set_title("Output Spikes")
ax4.plot(vm_mbon.t/ms, vm_mbon.v[0]/mV, label='mbon-1')
ax4.plot(vm_mbon.t/ms, vm_mbon.v[1]/mV, label='mbon-2')
ax5.eventplot(sup_spikes)
ax5.set_title("Supervision Spikes")
ax4.legend()

fig.suptitle(f"$N_\mathrm{{class}} = {N_CLASS}$")

plt.show()
