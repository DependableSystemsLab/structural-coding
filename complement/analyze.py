from analysis import sdc
from complement.parameters import SLURM_ARRAY
from storage import load, load_pickle


def draw_sdc():
    _, baseline = load_pickle('nonrecurring')
    print(sdc(baseline, baseline))
    for config in SLURM_ARRAY:
        faulty = load(config)
        print(config, sdc(baseline, faulty))


draw_sdc()