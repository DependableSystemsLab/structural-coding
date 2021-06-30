from collections import defaultdict

from analysis import sdc
from complement.parameters import SLURM_ARRAY, DEFAULTS
from storage import load, load_pickle


def draw_sdc():
    nonrecurring_pickle_name = 'nonrecurring_FashionMNISTTutorial'
    _, baseline, _ = load_pickle(nonrecurring_pickle_name)
    print(sdc(baseline, baseline))
    sdcs = defaultdict(list)
    missing = False
    for i, config in enumerate(SLURM_ARRAY):
        faulty = load(config, defaults=DEFAULTS)
        if faulty is None:
            print("{} in parameters array is missing.".format(i))
            missing = True
        else:
            sdcs[config['rank']].append(sdc(baseline, faulty))
    if missing:
        return
    for i in range(64):
        print(sum(j for j, _ in sdcs[i]) / len(sdcs[i]))


draw_sdc()
