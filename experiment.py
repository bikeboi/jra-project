import os
import shutil
from copy import deepcopy
import numpy as np
import matplotlib
import pynn_genn as sim
from pyNN.random import RandomDistribution
from pyNN.parameters import Sequence
from neo.io import PickleIO
from sacred import Experiment
from lib.embedding import spike_encode
from lib.data import get_inputs, separable
from lib.model import build_model, supervision, predicate
from lib.util import WeightLogger

matplotlib.use('GTK3Agg')

ex = Experiment()

@ex.config
def base_config():
    """Experiment Config"""
    # Input params
    n_class = 2
    n_epoch = 4
    downscale = 8
    input_transform = separable

    # Simulation parameters
    delta_t = 0.1
    t_snapshot = 50
    jitter = 5

    test_frac = 0.5
    sample_duration = n_class * 20 * t_snapshot
    test_duration = test_frac * sample_duration
    train_duration = (1 - test_frac) * sample_duration * n_epoch
    cooldown_duration = t_snapshot * 4

    duration = (
            train_duration  # Train period
            + cooldown_duration  # Cool down
            + test_duration  # Test period
            + t_snapshot  # Last interval
    )
    train_intervals = np.arange(0, train_duration, t_snapshot)
    test_intervals = np.arange(train_duration + cooldown_duration, duration - t_snapshot, t_snapshot)
    runs = 1
    model_type = 'unsupervised'
    tag = "muted"

    N_EKC = n_class
    INH = 100 / N_EKC
    A_PLUS = 0.01
    A_MINUS = 0.0

    # Model parameters
    model = dict(
        population={
            'n_pn': int(np.ceil(105 / downscale) ** 2),
            'n_ekc': N_EKC,
            's_ikc': 20
        },
        neuron={'tau_m': 5},
        synapse={
            0: {'weight': RandomDistribution('uniform', (5.0, 7.5)),
                'delay': 2.0},
            1: {'weight': 0.0,
                'delay': 5.0,
                'td': {
                    'A_minus': A_MINUS,
                    'A_plus': A_PLUS,
                    't_stop': train_duration,
                }},
        },
        swta=[-.75, -INH],  # 100 / neKC
        sparsity=0.1
    )


@ex.named_config
def unmuted_config():
    input_transform = None
    tag = "unmuted"


@ex.named_config
def multiclass():
    n_class = 3
    test_frac = 0.33


@ex.capture
def fetch_inputs(n_class, n_epoch, downscale, input_transform, test_frac):
    return get_inputs(
        "omniglot/python/images_background",
        n_class,
        n_epoch,
        downscale,
        test_frac=test_frac,
        transform=input_transform
    )


@ex.capture(prefix='model')
def construct_model(population, neuron, synapse, swta, sparsity):
    delta_t = sim.get_min_delay()
    return build_model(population['n_pn'], neuron, synapse, swta, delta_t, population['n_ekc'], population['s_ikc'],
                       sparsity)


# Helper
def prepare_simulation(model_type, tag, run, n_class, ekc_proj, t_snapshot):
    results_dir = f"results/{tag}_{model_type}_{n_class}"
    for pop in ["pn", "ikc", "ekc"]:
        shutil.rmtree(f"{results_dir}/data_{pop}_{run}.pickle", ignore_errors=True)
    os.makedirs(results_dir, exist_ok=True)
    weight_logger = WeightLogger(ekc_proj, t_snapshot, f"{results_dir}/weight_log.npy")
    return (
        PickleIO(f"{results_dir}/data_pn_{run}.pickle"),
        PickleIO(f"{results_dir}/data_ikc_{run}.pickle"),
        PickleIO(f"{results_dir}/data_ekc_{run}.pickle"),
        weight_logger
    )


@ex.automain
def main(n_class, model, N_EKC, INH, downscale, tag, model_type, delta_t, n_epoch, t_snapshot, jitter, train_duration,
         test_duration,
         test_frac, duration,
         cooldown_duration,
         train_intervals,
         test_intervals, runs):
    label_set = []

    for run_ix in range(runs):
        print("Run:", run_ix + 1)

        # Setup inputs
        sim.setup(delta_t)
        train_xs, train_ys, test_xs, test_ys = fetch_inputs()

        train_spikes = spike_encode(train_xs, train_ys,
                                    t_snapshot,
                                    spike_jitter=jitter)

        test_spikes = spike_encode(test_xs, test_ys,
                                   t_snapshot, offset=train_duration + cooldown_duration,
                                   spike_jitter=jitter)

        input_spikes = [Sequence(train + test) for train, test in zip(train_spikes, test_spikes)]

        # Set train spike times
        mb, projections = construct_model()
        mb.get_population('pn').record('spikes')
        mb.get_population('ekc').record('spikes')
        mb.get_population('ikc').record('spikes')
        mb.get_population('pn').set(spike_times=input_spikes)

        # Supervision
        if model_type == 'supervised':
            n = len(mb.get_population('ekc'))
            p = predicate.Predicate(lambda i, l, t: i == l)
            supervision\
                .Supervisor(p, n, off=0.0, on=5.0, t_duration=15.0, t_offset=5.0, until=train_duration)\
                .signal(train_ys, train_intervals)\
                .inject_into(mb.get_population('ekc'))

        mb.initialize(v=RandomDistribution('normal', (-65.0, 2.0)))

        # Preparations
        io_pn, io_ikc, io_ekc, weight_logger = prepare_simulation(model_type, tag, run_ix, n_class, projections[1],
                                                                  t_snapshot)

        # Run the simulation
        def logger(t):
            period = "Training" if t <= train_duration else "Testing"
            print(end='\r')
            print(f"{period}: Elapsed time:", sim.get_current_time(), "ms", end='')
            return t + t_snapshot

        print(f"Running simulation '{tag}_{model_type}' for", duration, "ms")
        sim.run(duration, callbacks=[logger, weight_logger])

        mb.get_population('ekc').write_data(io_ekc, 'spikes')
        mb.get_population('pn').write_data(io_pn, 'spikes')
        mb.get_population('ikc').write_data(io_ikc, 'spikes')

        weight_logger.save()

        label_set.append(test_ys)

        print()

        # Save weights for next epoch
        weights = projections[1].get('weight', format='array')
        sim.end()

    # Save data
    np.savez(
        f"results/{tag}_{model_type}_{n_class}_params.npz",
        runs=runs,
        train_intervals=train_intervals,
        test_intervals=test_intervals,
        duration=duration,
        t_snapshot=np.array(t_snapshot),
        labels=np.array(label_set),
        n_epoch=n_epoch,
        n_iKC=len(mb.get_population('ikc')),
        n_eKC=len(mb.get_population('ekc')),
    )
