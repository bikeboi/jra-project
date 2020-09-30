import numpy as np
import pynn_genn as sim
import lib.model.predicate as P


# Supervisor
class Supervisor:
    def __init__(self, predicate, n, on=1.0, off=-0.5, t_offset=0, t_duration=5.0, until=None):
        self.n = n
        if until:
            self.predicate = P.before(until) & predicate
        else:
            self.predicate = predicate
        self.on = on
        self.off = off
        self.t_offset = t_offset
        self.t_duration = t_duration

    def signal(self, labels, intervals) -> sim.StepCurrentSource:
        signal = []
        for i in range(self.n):
            on = np.array([(t + self.t_offset, self.on if self.predicate(i, lb, t) else self.off) for lb, t in
                           zip(labels, intervals)])
            off = on.copy()
            off[:, 0] += self.t_duration
            off[:, 1] = 0
            combined = np.array(sorted(np.concatenate([on, off], axis=0), key=lambda r: r[0]))
            signal.append(combined)

        signal = np.array(signal)
        times, amps = signal[:, :, 0].tolist(), signal[:, :, 1].tolist()

        return sim.StepCurrentSource(times=times, amplitudes=amps)


# Useful supervisors
class Full(Supervisor):
    def __init__(self, n, **kwargs):
        super().__init__(P.identity(), n, **kwargs)


class Asymmetric(Supervisor):
    def __init__(self, n, **kwargs):
        super().__init__(P.equal(0) & P.select(np.arange(int(n / 2))), n, **kwargs)


class Symmetric(Supervisor):
    def __init__(self, n, **kwargs):
        super().__init__(
            lambda i,l,t: (i < n/2 and l == 0) or (i >= n/2 and l == 1),
            #(P.identity() & P.select(np.arange(int(n / 2)))) | (~P.identity() & P.select(np.arange(int(n / 2), n))),
            n, **kwargs
        )

