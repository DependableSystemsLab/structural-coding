import os
from itertools import product


DOMAIN = {
    'model': ('vgg16', 'resnet50', 'googlenet', 'resnet101', 'resnet18', 'vgg19', 'alexnet', 'mobilenet_v3_large',
              'mobilenet_v3_small'),
    'pruning_method': ('global_unstructured',),
    'pruning_factor': (0., .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95),
    'inject': (True, False),
    'faults': (0, ),
}

CONSTRAINTS = (
    lambda c: sum(1 if c[k] else 0 for k in ('inject', 'faults', 'pruning_factor')) <= 1,
    lambda c: not c['inject'] and not c['pruning_factor'],
)

DEFAULTS = {}

SLURM_ARRAY = []

for combination in product(*DOMAIN.values()):
    c = {k: v for k, v in zip(DOMAIN.keys(), combination)}
    if all(constraint(c) for constraint in CONSTRAINTS):
        SLURM_ARRAY.append(c)

CONFIG = SLURM_ARRAY[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]


if __name__ == '__main__':

    print(len(SLURM_ARRAY))
