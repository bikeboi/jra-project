import numpy as np
import math
import pynn_genn as sim
from pyNN.parameters import Sequence
from ..cells import IF_curr_exp_adapt
from ..stdp import TemporarySpikePairRule


def build_populations(d_input, n_ekc, s_ikc, neuron_params):
    neuron = sim.IF_curr_exp(**neuron_params) #IF_curr_exp_adapt(**neuron_params)

    # Projection Neurons
    pn = sim.Population(d_input, sim.SpikeSourceArray(spike_times=[]), label="pn")

    # Intrinsic Kenyon Cells
    ikc = sim.Population(d_input * s_ikc, neuron, label="ikc")

    # Extrinsic Kenyon Cells
    ekc = sim.Population(n_ekc, neuron, label="ekc")

    return pn, ikc, ekc


def pn_to_ikc(pn, ikc, params, sparsity):
    return sim.Projection(
        pn, ikc,
        connector=sim.FixedProbabilityConnector(sparsity),
        synapse_type=sim.StaticSynapse(**params),
        receptor_type="excitatory",
        label="pn_ikc"
    )


def ikc_to_eKC(ikc, ekc, params, sparsity):
    return sim.Projection(
        ikc, ekc,
        connector=sim.FixedProbabilityConnector(sparsity),
        synapse_type=sim.STDPMechanism(
            TemporarySpikePairRule(**params['td']),
            sim.AdditiveWeightDependence(),
            weight=params['weight'],
            delay=params['delay']
        ),
        label="ikc_ekc"
    )


def sWTA(pop, weight, dt, proxy=False):
    if proxy:
        n = int(len(pop) * 0.1)
        neuron_params = {
            "tau_m": dt,
            "tau_syn_I": dt,
            "tau_syn_E": dt
        }

        proxy = sim.Population(n, sim.IF_curr_exp(**neuron_params))

        sim.Projection(pop, proxy, sim.AllToAllConnector(), sim.StaticSynapse(weight=1.0))
        sim.Projection(proxy, pop, sim.AllToAllConnector(), sim.StaticSynapse(weight=weight),
                       receptor_type="inhibitory")
    else:
        sim.Projection(pop, pop, sim.AllToAllConnector(allow_self_connections=False), sim.StaticSynapse(weight=weight),
                       receptor_type="inhibitory")


def build_model(d_input, neu_params, syn_params, swta, delta_t, n_ekc=100, s_ikc=10, sparsity=0.05):
    # Parameters
    neu = neu_params
    syn = syn_params

    # Populations
    neuron = IF_curr_exp_adapt(**neu)
    pn, ikc, ekc = build_populations(d_input, n_ekc, s_ikc, neu)

    # Projections
    proj_1 = pn_to_ikc(pn, ikc, syn[0], sparsity)
    proj_2 = ikc_to_eKC(ikc, ekc, syn[1], sparsity)
    sWTA(ikc, swta[0], delta_t, proxy=True)
    sWTA(ekc, swta[1], delta_t)

    model = sim.Assembly(pn, ikc, ekc)
    model.record("spikes")

    return model, [proj_1, proj_2]


# Utility
def num_divergent_conn(p, n_to):
    return int(p * n_to)


def num_convergent_conn(p, n_from):
    return int(p * n_from)
