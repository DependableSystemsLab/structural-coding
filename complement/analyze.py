from collections import defaultdict

from analysis import sdc
from complement.parameters import SLURM_ARRAY, DEFAULTS
from storage import load, load_pickle


def draw_sdc():
    _, baseline, _ = load_pickle('nonrecurring')
    print(sdc(baseline, baseline))
    sdcs = defaultdict(list)
    for config in SLURM_ARRAY:
        faulty = load(config, defaults=DEFAULTS)
        sdcs[config['rank']].append(sdc(baseline, faulty))
    for i in range(64):
        print(sum(j for j, _ in sdcs[i]) / len(sdcs[i]))


draw_sdc()
