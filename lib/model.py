import numpy as np
import math
import pynn_genn as sim
from pyNN.random import NumpyRNG
from pyNN.random import RandomDistribution
from pyNN.standardmodels import build_translations


def default_MB(inputs, **params):
    # Populations
    p_params = params['population']

    pop_PN = build_PN(inputs)
    pop_iKC = build_population(p_params['n_iKC'], p_params['neuron_iKC'], "iKC")
    pop_eKC = build_population(p_params['n_eKC'], p_params['neuron_eKC'], "eKC")

    pop_PN, pop_iKC, pop_eKC = default_populations(inputs, **params['population'])

    # Projections
    proj_PN_iKC = default_proj_PN_iKC(pop_PN, pop_iKC, **params['syn_PN_iKC'])
    proj_iKC_eKC = default_proj_iKC_eKC(pop_iKC, pop_eKC, **params['syn_iKC_eKC'])

    return {
        "model": sim.Assembly(pop_PN, pop_iKC, pop_eKC),
        "projections": { 
            "PN_iKC": proj_PN_iKC,
            "iKC_eKC": proj_iKC_eKC
        }
    }


def default_populations(input_spikes, **params):
    PN = build_PN(input_spikes)
    iKC = build_population(params['n_iKC'], params['neuron_iKC'], "iKC")
    eKC = build_population(params['n_eKC'], params['neuron_eKC'], "eKC")

    return PN, iKC, eKC


def default_proj_PN_iKC(PN, iKC, **params):
    return sim.Projection(
        PN, iKC,
        params['conn'],
        sim.StaticSynapse(weight=params['g'], delay=params['tau']),
        label="PN->iKC"
    )


def default_proj_iKC_eKC(iKC, eKC, **params):
    conn = [ 
        ( np.random.randint(0,len(eKC)), np.random.randint(0,len(iKC)), params['g'].next(), params['tau']) for i in range(int(len(eKC) * params['p']))
    ]

    for c in conn:
        print(c)

    proj = sim.Projection(
        iKC, eKC,
        sim.FixedProbabilityConnector(params['p']), #sim.FromListConnector(conn, column_names=['weight', 'delay']),
        sim.STDPMechanism(
            weight=params['g'].next(),
            delay=params['tau'],
            timing_dependence=params['TD'],
            weight_dependence=params['WD'],
        ),
        label="iKC->eKC"
    )

    print(proj.shape)

    return proj


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



## Mushroom Body parameters
def default_params(
    delta_t, t_snapshot, d_input, n_samples, n_eKC, s_iKC=20, test_frac=0.3, rng=NumpyRNG(),
    neuron_iKC={}, neuron_eKC={}, syn_PN_iKC={}, syn_iKC_eKC={}
    ):

    """
    Generate simulation and model parameters
    """

    steps = (
        t_snapshot # Initial pause
        + t_snapshot * n_samples # Input period 1
        #+ t_snapshot * 4 # Cooling period
        #+ t_snapshot * n_samples # Input period 2
        + t_snapshot # Final pause
    )

    neuron_iKC = sim.IF_curr_exp() if not neuron_iKC else neuron_iKC
    neuron_eKC = sim.IF_curr_exp() if not neuron_eKC else neuron_eKC

    PN_iKC = PARAM_PN_iKC(rng)
    iKC_eKC = PARAM_iKC_eKC(rng)

    PN_iKC.update(syn_PN_iKC)
    iKC_eKC.update(syn_iKC_eKC)

    return {
        # Population parameters
        "population": {
            "n_iKC": s_iKC * d_input,
            "n_eKC": n_eKC,
            "neuron_iKC": neuron_iKC,
            "neuron_eKC": neuron_eKC
        },

        # Synapse parameters
        "syn_PN_iKC": PN_iKC,

        # Plasticity parameters
        "syn_iKC_eKC": iKC_eKC,

        # Simulation parameters
        "steps": steps,
        "delta_t": delta_t,
        "t_snapshot": t_snapshot,
        "n_sample": n_samples,
        "rng": rng,
        "snapshot_noise": 0.0
    }

# Defaults
PARAM_PN_iKC = lambda rng=None: ({
    "conn": sim.FixedProbabilityConnector(0.5),
    "g": RandomDistribution('normal', (5.0, 1.25), rng=rng),
    "tau": 2.0,
})

PARAM_iKC_eKC = lambda rng=None: ({
    "p": 0.2,
    "g": RandomDistribution('normal', (0.125, 0.1), rng=rng),
    "tau": 10.0,

    "WD": sim.AdditiveWeightDependence(),
    "TD": sim.SpikePairRule(),
})