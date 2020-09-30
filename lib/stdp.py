from copy import deepcopy
from pyNN.standardmodels import synapses, StandardModelType, build_translations
from pynn_genn.simulator import state
import numpy as np
import logging
from pygenn.genn_wrapper.WeightUpdateModels import StaticPulse
from pynn_genn.model import GeNNStandardSynapseType, GeNNDefinitions
from pynn_genn.standardmodels.synapses import WeightDependence, delayMsToSteps, delayStepsToMs, DDTemplate


class TemporarySpikePairRule(synapses.STDPTimingDependence):
    # __doc__ = synapses.SpikePairRule.__doc__

    vars = {"tauPlus": "scalar",  # 0 - Potentiation time constant (ms)
            "tauMinus": "scalar", # 1 - Depression time constant (ms)
            "Aplus": "scalar",    # 2 - Rate of potentiation
            "Aminus": "scalar",   # 3 - Rate of depression
            "Tstop": "scalar"}

    pre_var_name_types = [("preTrace", "scalar")]
    post_var_name_types = [("postTrace", "scalar")]

    # using {.brc} for left{ or right} so that .format() does not freak out
    sim_code = DDTemplate("""
        if (dt > 0 && $(Tstop) > 0 && $(t) < $(Tstop))
        {
            const scalar update = $(Aminus) * $(postTrace) * exp(-dt / $(tauMinus));
            $${WD_CODE}
        }
        """)

    learn_post_code = DDTemplate("""
        if (dt > 0 && $(Tstop) > 0 && $(t) < $(Tstop))
        {
            const scalar update = $(Aplus) * $(preTrace) * exp(-dt / $(tauPlus));
            $${WD_CODE}
        }
        """)

    pre_spike_code = """\
        const scalar dt = $(t) - $(sT_pre);
        $(preTrace) = $(preTrace) * exp(-dt / $(tauPlus)) + 1.0;
        """

    post_spike_code = """\
        const scalar dt = $(t) - $(sT_post);
        $(postTrace) = $(postTrace) * exp(-dt / $(tauMinus)) + 1.0;
        """

    translations = build_translations(
        ("tau_plus",   "tauPlus"),
        ("tau_minus",  "tauMinus"),
        ("A_plus",     "Aplus"),
        ("A_minus",    "Aminus"),
        ("t_stop",     "Tstop")
    )

    default_parameters = {
        'tau_plus': 20.0,
        'tau_minus': 20.0,
        'A_plus': 0.01,
        'A_minus': 0.0,
        't_stop': -1.0,
    }

    def __init__(self, A_plus=0.01, A_minus=0.01, tau_plus=20.0, tau_minus=20.0, t_stop=-1):
        """
        Create a new specification for the timing-dependence of an STDP rule.
        """
        parameters = dict(locals())
        parameters.pop('self')
        if np.isinf(t_stop):
            parameters['t_stop'] = -1.0

        synapses.STDPTimingDependence.__init__(self, **parameters)
