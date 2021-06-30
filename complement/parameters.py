import os
from itertools import product


DOMAIN = {
    'model': ('resnet50', 'FashionMNISTTutorial'),
    'rank': tuple(range(64)),
    'bit_position': tuple(range(23, 32)),
    'protection': ('none', 'clipper'),
    'ranking': ('gradient', 'random')
}

CONSTRAINTS = (
    lambda c: c['model'] == 'FashionMNISTTutorial',
)

DEFAULTS = {
    'ranking': 'gradient'
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

