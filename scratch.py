# Sandbox to test out implementations


from models.stdp import gary_stdp
import matplotlib.pyplot as plt
import brian2.numpy_ as np
from non_overlapping.util import get_trains
from brian2.only import *

plt.switch_backend("GTK3Agg")

start_scope()

duration = 30
n_pre = 50


n1 = SpikeGeneratorGroup(1, np.zeros(20), np.linspace(5, 30, 20) * ms)
n2 = SpikeGeneratorGroup(1, np.zeros(1), np.array([15]) * ms)

s = Synapses(n1, n2, gary_stdp.model(),
             on_pre=gary_stdp.on_pre(),
             on_post=gary_stdp.on_post())
s.connect()
s.tpre = -np.inf * ms
s.tpost = -np.inf * ms
s.aplus = 0.5
s.aminus = 0.1

spm1 = SpikeMonitor(n1)
spm2 = SpikeMonitor(n2)

sm_w = StateMonitor(s, ['tpre', 'tpost', 'w'], record=True)

run(duration * ms)

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.set_title("Pre Neuron")
ax1.plot(spm1.t / ms, spm1.i, '.')

ax2.set_title("Post Neuron")
ax2.plot(spm2.t / ms, spm2.i, '.')

#ax3.plot(sm_w.t / ms, sm_w.tpre[0]/ms, label="pre spike time")
#ax3.plot(sm_w.t / ms, sm_w.tpost[0]/ms, label="post spike time")
ax3.plot(sm_w.t / ms, sm_w.w[0], label="w")

ax3.legend()

plt.show()
