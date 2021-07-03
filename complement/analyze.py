from collections import defaultdict

from analysis import sdc
from complement.parameters import SLURM_ARRAY, DEFAULTS
from storage import load, load_pickle


def draw_sdc():
    baselines = {}
    _, baselines['FashionMNISTTutorial'] = load_pickle('nonrecurring_FashionMNISTTutorial')[:2]
    _, baselines['FashionMNISTTutorial_smooth'] = load_pickle('nonrecurring_FashionMNISTTutorial_smooth')[:2]
    _, baselines['resnet50'] = load_pickle('nonrecurring_resnet50')[:2]
    sdcs = defaultdict(list)
    key = lambda c: (c['model'], c['protection'], c['ranking'])
    missing = False
    percent = len(SLURM_ARRAY) // 100
    for i, config in enumerate(SLURM_ARRAY):
        faulty = load(config, defaults=DEFAULTS)
        if faulty is None:
            print("{} in parameters array is missing.".format(i))
            missing = True
        else:
            sdcs[key(config)].append(sdc(baselines[config['model']], faulty))
        if i % percent == 0:
            print(i // percent, '%')
    if missing:
        return
    for i in sdcs:
        print(i, sum(j for j, _ in sdcs[i]) / len(sdcs[i]))


draw_sdc()
