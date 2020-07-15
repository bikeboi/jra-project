import numpy as np
import math
import pynn_genn as sim
from pyNN.random import RandomDistribution
from pyNN.standardmodels import build_translations


def default_MB(inputs, **params):
    # Populations
    p_params = params['population']

    pop_PN = build_PN(inputs)
    pop_iKC = build_population(p_params['n_iKC'], params['neuron'], "iKC")
    pop_eKC = build_population(p_params['n_eKC'], params['neuron'], "eKC")

    # Connections
    s_params = params['synapse']

    ## PNs to iKCs
    conn_PN_iKC = sim.FixedProbabilityConnector(s_params['p_PN_iKC'], rng=params['rng'])
    proj_PN_iKC = sim.Projection(
        pop_PN, pop_iKC,
        conn_PN_iKC,
        synapse_type=sim.StaticSynapse(weight=s_params['g_PN_iKC'], delay=s_params['t_PN_iKC']),
        receptor_type='excitatory',
        label="PN->iKC"
    )

    ## iKCs to eKCs
    conn_iKC_eKC = sim.FixedProbabilityConnector(s_params['p_iKC_eKC'], rng=params['rng'])
    proj_iKC_eKC = sim.Projection(
        pop_iKC, pop_eKC,
        conn_iKC_eKC,
        synapse_type=sim.STDPMechanism(**params['plasticity'], ),
        receptor_type='excitatory',
        label="iKC->eKC"
    )

    return {
        "model": sim.Assembly(pop_PN, pop_iKC, pop_eKC),
        "projections": { 
            "PN_iKC": proj_PN_iKC,
            "iKC_eKC": proj_iKC_eKC
        }
    }


def build_PN(spike_times) -> sim.Population:
    population = sim.Population(
        len(spike_times),
        sim.SpikeSourceArray(spike_times=spike_times),
        label="PN"
    )

    return population


def build_population(n, neuron_type, label) -> sim.Population:
    return sim.Population(
        n, neuron_type, label=label
    )
