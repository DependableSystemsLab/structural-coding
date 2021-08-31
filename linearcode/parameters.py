import os
from itertools import product


DOMAIN = {
    'model': ('resnet50', ),
    'injection': range(4000),
    # 'injection': range(400),
    'protection': ('none', 'clipper', 'sc'),
    'sampler': ('none', 'critical'),
    'flips': (2, 4, 8, 16, 32, )
}

CONSTRAINTS = (
    lambda c: c['sampler'] == 'critical',
    lambda c: c['protection'] == 'sc',
    lambda c: c['flips'] == 2,
)

DEFAULTS = {
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

