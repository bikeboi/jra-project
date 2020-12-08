"""STDP Mechanism Model Equations"""

from brian2 import Equations


class STDPModel:
    """Wrapper around STDP model and pre-post equations"""

    def __init__(self, equation, on_pre="", on_post=""):
        self.equation = equation
        self.pre = on_pre
        self.post = on_post

    def model(self, **params):
        return Equations(self.equation, **params)

    def on_pre(self):
        return self.pre

    def on_post(self):
        return self.post


# Models
# 'Standard' STDP
standard_stdp = STDPModel(
    """
    dapre/dt = apre/taupre
    """,

    on_pre="""
    """,

    on_post="""
    """,
)

# Gary's Model
gary_stdp = STDPModel(
    """
    g : 1
    tpre : second
    tpost : second
    aplus : 1
    aminus : 1
    tplus : second
    tminus : second
    gmin : 1
    gmax : 1
    """,

    # Update pre-synaptic spike time and update weight
    on_pre="""
    v += g*mV
    tpre = t
    g = clip(g + aplus*int(abs(tpost - tpre) <= tplus) - aminus*int(abs(tpost-tpre) > tplus and abs(tpost-tpre) <= tminus), gmin, gmax)
    """,

    # Update post-synaptic spike time and update weight
    on_post="""
    tpost = t
    g = clip(g + aplus*int(abs(tpost - tpre) <= tplus) - aminus*int(abs(tpost-tpre) > tplus and abs(tpost-tpre) <= tminus), gmin, gmax)
    """,
)
