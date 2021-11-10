import os
from copy import deepcopy
from itertools import product

from settings import PROBABILITIES

DOMAIN = {
    'injection': range(400),
    # 'injection': range(1),
    'model': ('e2e', 'resnet50', 'alexnet', 'squeezenet', 'vgg19', 'mobilenet', 'googlenet', 'shufflenet'),
    'quantization': (True, False),
    'sampler': ('none', 'critical', 'tiny'),
    'dataset': ('full_imagenet', 'imagenet', 'imagenet_test',
                # 10,000 deployment images as used in "Optimizing Selective Protection for CNN Resilience"
                'imagenet_ds',
                # a 128 (4 batches) sub sample
                'imagenet_ds_128',
                'comma.ai', 'commaai_test',
                'driving_dataset_test'),
    'flips': (1,
              # 2,
              4,
              # 8,
              16,
              # 32,
              0,
              *PROBABILITIES,
              'word',
              'column',
              'row',
              # 'bank',
              # 'chip',
              'rowhammer'),
    'protection': ('none', 'clipper', 'ranger', 'sc', 'radar', 'milr', 'flr_mr', 'tmr'),
}

# don't use short circuit execution here
CONSTRAINTS = (
    lambda c: c['dataset'] in ('imagenet_ds_128', 'driving_dataset_test'),
    lambda c: any((
        all((c['dataset'] == 'imagenet_ds_128', c['model'] != 'e2e')),
        all((c['dataset'] == 'driving_dataset_test', c['model'] == 'e2e'))
    )),
    lambda c: c['sampler'] == 'none',
    lambda c: any((c['flips'] != 0, all((c['injection'] == 0, c['protection'] == 'none')))),
    # ensure baseline execution
    lambda c: c['protection'] in ('sc', 'none', 'clipper', 'tmr', 'radar', 'milr', 'ranger'),
    lambda c: c['model'] not in ('vgg19', ),
    lambda c: not c['quantization'],

    # retry
    lambda c: any((
        not isinstance(c['flips'], str),
        c['model'] == 'e2e',
    )),

    # retry
    lambda c: any((
        all((c['flips'] != 0, isinstance(c['flips'], int))),
        all((c['model'] == 'e2e', c['protection'] == 'radar')),
    )),
)

DEFAULTS = {
    'sampler': 'none',
    'dataset': 'imagenet',
    'quantization': False,
}


def get_constraint_keys(cstrnt):
    class GetItemObserver:
        def __init__(self):
            self.observed = set()

        def __getitem__(self, item):
            self.observed.add(item)
            return DOMAIN[item][0]

    observer = GetItemObserver()
    cstrnt(observer)
    return observer.observed


def constrain_domain(full_domain, constraints):
    constrained_domain = deepcopy(full_domain)
    for constraint in constraints:
        constraint_keys = get_constraint_keys(constraint)
        if len(constraint_keys) == 1:
            domain = next(iter(constraint_keys))
            constrained_domain[domain] = tuple(v for v in constrained_domain[domain] if constraint({domain: v}))
    return constrained_domain


def query_configs(constraints, domain=None):
    if domain is None:
        domain = DOMAIN
    domain = constrain_domain(domain, constraints)
    result = []
    for combination in product(*domain.values()):
        c = {k: v for k, v in zip(domain.keys(), combination)}
        if all(constraint(c) for constraint in constraints):
            result.append(c)
    return result


SLURM_ARRAY = query_configs(CONSTRAINTS)

CONFIG = SLURM_ARRAY[int(os.environ.get('INTERNAL_SLURM_ARRAY_TASK_ID'))]

if __name__ == '__main__':
    print(len(SLURM_ARRAY), len(SLURM_ARRAY) / 40)
