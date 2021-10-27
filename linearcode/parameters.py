import os
from copy import deepcopy
from itertools import product

from settings import PROBABILITIES

DOMAIN = {
    'injection': range(400),
    'model': ('resnet50', 'alexnet', 'squeezenet', 'vgg19', 'mobilenet', 'googlenet', 'shufflenet', 'e2e'),
    'quantization': (True, False),
    'protection': ('none', 'clipper', 'ranger', 'sc', 'radar', 'milr', 'flr_mr', 'tmr'),
    'sampler': ('none', 'critical', 'tiny'),
    'dataset': ('full_imagenet', 'imagenet', 'imagenet_test',
                # 10,000 deployment images as used in "Optimizing Selective Protection for CNN Resilience"
                'imagenet_ds',
                # a 128 (4 batches) sub sample
                'imagenet_ds_128',
                'comma.ai', 'commaai_test'),
    'flips': (1, 2, 4, 8, 16, 32,
              0,
              *PROBABILITIES,
              'word',
              'column',
              'row',
              'bank',
              'chip',
              'rowhammer-0.0003')
}

# don't use short circuit execution here
CONSTRAINTS = (
    lambda c: c['dataset'] == 'imagenet_ds_128',
    lambda c: c['sampler'] == 'none',
    lambda c: any((c['flips'] != 0, all((c['injection'] == 0, c['protection'] == 'none')))),
    # ensure baseline execution
    lambda c: c['protection'] in ('sc', 'none', 'clipper', 'tmr', 'radar'),
    lambda c: c['flips'] < 1,
    lambda c: c['model'] not in ("e2e", 'vgg19'),
    # lambda c: not c['quantization'],
    lambda c: c['quantization'],

    # test quantized
    lambda c: c['protection'] in ('sc', 'radar', 'none'),
    lambda c: c['injection'] < 2,

    # this
    # lambda c: c['model'] == 'shufflenet',
    # lambda c: c['protection'] == 'sc',
    # lambda c: c['flips'] == 0.00000552972,
)
#
# CONSTRAINTS = (
#     lambda c: c['sampler'] == 'tiny',
#     lambda c: c['protection'] in ('sc', 'none', 'clipper'),
#     lambda c: c['flips'] <= 1,
#     lambda c: not (c['model'] == 'alexnet' and c['protection'] == 'clipper'),
#     lambda c: c['injection'] == 1,
# )

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
