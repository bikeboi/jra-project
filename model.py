import numpy as np
import math
import pynn_genn as sim
from pyNN.random import RandomDistribution
from pyNN.standardmodels import build_translations

from embedding import spatiotemporal

def build_PN(spike_times) -> sim.Population:
    population = sim.Population(
        len(spike_times),
        sim.SpikeSourceArray(spike_times=spike_times),
        label="PN"
    )

    return population


def build_LHI(n, neuron_type) -> sim.Population:
    population = sim.Population(
        n,
        neuron_type,
        label="LHI"
    )

    return population


def build_iKC(n, neuron_type) -> sim.Population:
    population = sim.Population(
        n,
        neuron_type,
        label="iKC"
    )

    return population


def build_eKC(n, neuron_type) -> sim.Population:
    population = sim.Population(
        n,
        neuron_type,
        label="eKC"
    )
    
    return population


def build_MB(input_spikes, **params):
    # Populations
    p_params = params['population']

    pop_PN = build_PN(input_spikes)
    pop_LHI = build_LHI(p_params['n_LHI'], p_params['neuron_type'])
    pop_iKC = build_iKC(p_params['n_iKC'], p_params['neuron_type'])
    pop_eKC = build_eKC(p_params['n_eKC'], p_params['neuron_type'])

    # Connections
    s_params = params['synapse']

    # PNs to iKCs
    conn_PN_iKC = sim.FixedProbabilityConnector(s_params['p_PN_iKC'], rng=params['rng'])
    syn_PN_iKC = sim.StaticSynapse(
        weight=s_params['g_PN_iKC'],
        delay=s_params['t_PN_iKC']
    )
    proj_PN_iKC = sim.Projection(
        pop_PN, pop_iKC,
        conn_PN_iKC,
        synapse_type=syn_PN_iKC,
        receptor_type='excitatory',
        label="PN->iKC"
    )

    # LHI Gain Control
    sim.Projection( # PN -> LHI
        pop_PN, pop_LHI,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=(53.75/(params['n_input'] + 15)), delay=1.0),
        receptor_type='excitatory',
        label="PN->LHI"
    )

    sim.Projection( # LHI -> iKC
        pop_LHI, pop_iKC,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=s_params['g_LHI_iKC'], delay=s_params['t_LHI_iKC']),
        receptor_type='inhibitory',
        label="LHI->iKC"
    )

    # iKCs to eKCs
    conn_iKC_eKC = sim.FixedProbabilityConnector(s_params['p_iKC_eKC'], rng=params['rng'])

    proj_iKC_eKC = sim.Projection(
        pop_iKC, pop_eKC,
        conn_iKC_eKC,
        synapse_type=stdp_model(
            params['plasticity'],
            s_params['g_iKC_eKC'],
            s_params['t_iKC_eKC']
        ),
        receptor_type='excitatory',
        label="iKC->eKC"
    )

    # Lateral inhibition in eKCs
    sim.Projection(
        pop_eKC, pop_eKC,
        sim.AllToAllConnector(allow_self_connections=False),
        synapse_type=sim.StaticSynapse(
            weight=s_params['g_eKC_inh'], 
            delay=s_params['t_eKC_inh']
        ),
        receptor_type='inhibitory',
        label="eKC->eKC"
    )


    # Assemble and return
    return sim.Assembly(pop_PN, pop_LHI, pop_iKC, pop_eKC), proj_PN_iKC, proj_iKC_eKC

# Helpers
def stdp_model(params, weight, delay):
    # Parameters
    c_10, c_01, c_11 = params['c_10'], params['c_01'], params['c_11']
    t_l = params['t_l']
    g_0, g_max = params['g_0'], params['g_max']

    # Derived parameters
    tau_plus = 20 #(1/c_01+1/c_01)*t_l*c_11/2
    tau_minus = 20#-(1/c_10+1/c_11)*t_l*c_11/2

    A_plus = 0.01#2*g_max/(t_l*c_11)
    A_minus = 0.012#-A_plus

    # Timing dependence
    time_dep = sim.SpikePairRule(
        tau_plus, tau_minus, 
        A_plus, A_minus
    )

    # Weight dependence
    weight_dep = sim.MultiplicativeWeightDependence(
        w_min=g_0, w_max=g_max,
    )

    return sim.STDPMechanism(
        weight=weight,
        delay=delay,

        weight_dependence=weight_dep,
        timing_dependence=time_dep
    )

# Experiment with this later
"""
class EKCWeightDependence(sim.MultiplicativeWeightDependence, sim.WeightDependence):
    __doc__ = sim.MultiplicativeWeightDependence.__doc__

    depression_update_code = "$(g) -= ($(g) - $(Wmin)) * update;\n"

    potentiation_update_code = "$(g) += ($(Wmax) - $(g)) * update;\n"

    translations = build_translations(*sim.WeightDependence.wd_translations)
"""