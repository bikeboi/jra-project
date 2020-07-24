from pynn_genn.model import GeNNDefinitions, GeNNStandardCellType
from pyNN.standardmodels import cells, build_translations
from pynn_genn.standardmodels.cells import genn_postsyn_defs, tau_to_decay, tau_to_init
from functools import partial


_genn_neuron_defs = {}

_genn_neuron_defs['IFAdapt'] = GeNNDefinitions(

    definitions={
        "sim_code": """
            $(I) = $(Isyn);
            if ($(RefracTime) <= 0.0) {
                scalar alpha = ( ($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);
                $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
                $(VThreshAdapt) = $(Vthresh) + ($(VThreshAdapt) - $(Vthresh)) * $(ThreshDecay);
            } else {
                $(RefracTime) -= DT;
            }
        """,

        "threshold_condition_code": "$(RefracTime) <= 0.0 && $(V) >= $(VThreshAdapt)",

        "reset_code": """
            $(V) = $(Vreset);
            $(RefracTime) = $(TauRefrac);
            $(VThreshAdapt) += $(UpThresh) * ( $(Vthresh) - $(Vrest) );
        """,

        "var_name_types": [
            ("V", "scalar"),
            ("I", "scalar"),
            ("RefracTime", "scalar"),
            ("VThreshAdapt", "scalar")
        ],

        "param_name_types": {
            "Rmembrane": "scalar",
            "ExpTC": "scalar",
            "Vrest": "scalar",
            "Vreset": "scalar",
            "Vthresh": "scalar",
            "Ioffset": "scalar",
            "TauRefrac": "scalar",
            "UpThresh": "scalar",
            "ThreshDecay": "scalar",
        }
    },

    translations=(
        ("v_rest",      "Vrest"),
        ("v_reset",     "Vreset"),
        ("cm",          "Rmembrane",     "tau_m / cm", ""),
        ("tau_m",       "ExpTC",         partial(tau_to_decay, "tau_m"), None),
        ("tau_refrac",  "TauRefrac"),
        ("v_threshold",    "Vthresh"),
        ("i_offset",    "Ioffset"),
        ("v",           "V"),
        ("i",           "I"),
        ("w_threshold", "UpThresh"),
        ("tau_threshold",  "ThreshDecay",    partial(tau_to_decay, "tau_threshold"), None),
        ("v_thresh_adapt",    "VThreshAdapt"),
    ),

    extra_param_values={
        "RefracTime": 0.0,
    }
)

class IF_curr_exp_adapt(cells.IF_curr_exp, GeNNStandardCellType):
    __doc__ = cells.IF_curr_exp.__doc__

    default_parameters = {
        'v_rest': -65.0,  # Resting membrane potential in mV.
        'cm': 1.0,  # Capacity of the membrane in nF
        'tau_m': 20.0,  # Membrane time constant in ms.
        'tau_refrac': 0.1,  # Duration of refractory period in ms.
        'tau_syn_E': 5.0,  # Decay time of excitatory synaptic current in ms.
        'tau_syn_I': 5.0,  # Decay time of inhibitory synaptic current in ms.
        'i_offset': 0.0,  # Offset current in nA
        'v_reset': -65.0,  # Reset potential after a spike in mV.
        'v_threshold': -50.0,  # Spike threshold in mV. STATIC, MIN
        'i': 0.0, #nA total input current
        'tau_threshold': 120.0,
        'w_threshold': 1.8,
        'v_thresh_adapt': -50.0,  # Spike threshold in mV.

    }

    recordable = ['spikes', 'v', 'i', 'v_thresh_adapt']

    default_initial_values = {
        'v': -65.0,  # 'v_rest',
        'isyn_exc': 0.0,
        'isyn_inh': 0.0,
        'i': 0.0,
    }

    units = {
        'v': 'mV',
        'isyn_exc': 'nA',
        'isyn_inh': 'nA',
        'v_rest': 'mV',
        'cm': 'nF',
        'tau_m': 'ms',
        'tau_refrac': 'ms',
        'tau_syn_E': 'ms',
        'tau_syn_I': 'ms',
        'i_offset': 'nA',
        'v_reset': 'mV',
        'v_threshold': 'mV',
        'i': 'nA',
        'tau_threshold': 'ms',
        'w_threshold': '',
        'v_thresh_adapt': 'mV',
    }

    receptor_types = (
        'excitatory', 'inhibitory',
    )

    genn_neuron_name = "IFAdapt"
    genn_postsyn_name = "ExpCurr"
    neuron_defs = _genn_neuron_defs['IFAdapt']
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]