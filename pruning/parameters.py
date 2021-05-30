import os
from copy import copy
from itertools import product


DOMAIN = {
    'model': ('vgg16', 'resnet50',
              # 'googlenet',  commented out because of not using ReLU
              'resnet101', 'resnet18', 'vgg19', 'alexnet', 'mobilenet_v3_large',
              'mobilenet_v3_small'),
    'pruning_method': ('global_unstructured',),
    'pruning_factor': (0., .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95),
    'inject': (False, True),
    'faults': (0, 10),
    'protection': ('none', 'clipper')
}

CONSTRAINTS = (
    lambda c: c['inject'],
    lambda c: c['faults'] == 10,
    lambda c: c['model'] in ('vgg16', 'resnet50')
)

DEFAULTS = {
    'protection': 'none'
}

BASELINE_CONFIG = {k: DOMAIN[k][0] for k in DOMAIN}

SLURM_ARRAY = []

for combination in product(*DOMAIN.values()):
    c = {k: v for k, v in zip(DOMAIN.keys(), combination)}
    if all(constraint(c) for constraint in CONSTRAINTS):
        SLURM_ARRAY.append(c)

CONFIG = SLURM_ARRAY[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]


if __name__ == '__main__':
    print(len(SLURM_ARRAY))

