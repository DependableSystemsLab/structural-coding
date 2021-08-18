import os
from itertools import product


DOMAIN = {
    'model': ('resnet50', 'FashionMNISTTutorial', 'FashionMNISTTutorial_smooth'),
    'rank': tuple(range(4000)),
    'bit_position': tuple(range(23, 32)),
    'protection': ('none', 'clipper', 'ranger', 'radar'),
    'ranking': ('gradient', 'random', 'gradient_protected_20'),
    'sampler': ('none', 'critical'),
}

CONSTRAINTS = (
    lambda c: c['model'] == 'resnet50',
    lambda c: c['ranking'] == 'random',
    lambda c: c['sampler'] == 'critical',
)

DEFAULTS = {
    'ranking': 'gradient',
    'sampler': 'none',
}

BASELINE_CONFIG = {k: DOMAIN[k][0] for k in DOMAIN}

SLURM_ARRAY = []

for combination in product(*DOMAIN.values()):
    c = {k: v for k, v in zip(DOMAIN.keys(), combination)}
    if all(constraint(c) for constraint in CONSTRAINTS):
        SLURM_ARRAY.append(c)

CONFIG = SLURM_ARRAY[int(os.environ.get('INTERNAL_SLURM_ARRAY_TASK_ID'))]


if __name__ == '__main__':
    print(len(SLURM_ARRAY))

