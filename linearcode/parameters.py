import os
import pickle
from copy import deepcopy
from itertools import product

from settings import PROBABILITIES, SHARD, INJECTIONS_RANGE
from storage import get_storage_filename

DOMAIN = {
    'injection': range(*INJECTIONS_RANGE),
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
              'row-4',
              # 'bank',
              # 'chip',
              # 'flr',
              'rowhammer'),
    'protection': ('none', 'clipper', 'ranger', 'sc', 'radar', 'milr', 'flr_mr', 'tmr', 'opt', 'secded', 'chipkill'),
}

# don't use short circuit execution here
artifact_description_base = (
    lambda c: c['dataset'] == 'imagenet_ds_128',
    lambda c: c['sampler'] in ('none', 'tiny'),
    # ensure baseline execution
    lambda c: any((c['flips'] != 0, c['injection'] == 0)),
    # only baseline
    lambda c: isinstance(c['flips'], str) or isinstance(c['flips'], float) or c['flips'] == 0,
    lambda c: c['protection'] in ('sc', 'none', 'clipper', 'radar', 'milr', 'ranger', 'opt', 'chipkill'),
    lambda c: c['model'] not in ('vgg19', 'e2e'),
    lambda c: not c['quantization'],
    lambda c: c['flips'] not in ('rowhammer', 'row-4'),
    # limit chipkill and opt
    lambda c: any((isinstance(c['flips'], float), c['protection'] != 'opt')),
    lambda c: any((c['flips'] == PROBABILITIES[0], c['protection'] != 'chipkill')),
)
SHARDS_CONSTRAINTS = {
    'default': (
        lambda c: c['dataset'] in ('imagenet_ds_128', 'driving_dataset_test', 'imagenet'),
        lambda c: any((
            all((c['dataset'] != 'driving_dataset_test', c['model'] != 'e2e')),
            all((c['dataset'] == 'driving_dataset_test', c['model'] == 'e2e'))
        )),
        lambda c: c['sampler'] in ('none', 'tiny'),
        # ensure baseline execution
        lambda c: any((c['flips'] != 0, c['injection'] == 0)),
        # only baseline
        lambda c: isinstance(c['flips'], str) or isinstance(c['flips'], float) or c['flips'] == 0,
        lambda c: c['protection'] in ('sc', 'none', 'clipper', 'tmr', 'radar', 'milr', 'ranger', 'opt', 'secded', 'chipkill'),
        lambda c: c['model'] not in ('vgg19',),
        lambda c: not c['quantization'],
    ),
    'missing': (
        lambda c: c['dataset'] in ('imagenet_ds_128', 'driving_dataset_test'),
        lambda c: any((
            all((c['dataset'] == 'imagenet_ds_128', c['model'] != 'e2e')),
            all((c['dataset'] == 'driving_dataset_test', c['model'] == 'e2e'))
        )),
        lambda c: c['sampler'] == 'none',
        # ensure baseline execution
        lambda c: any((c['flips'] != 0, all((c['injection'] == 0, c['protection'] == 'none')))),
        # only baseline
        lambda c: isinstance(c['flips'], str) or isinstance(c['flips'], float) or c['flips'] == 0,
        lambda c: c['protection'] in ('sc', 'none', 'clipper', 'tmr', 'radar', 'milr', 'ranger'),
        lambda c: c['model'] not in ('vgg19',),
        lambda c: not c['quantization'],

        # retry
        lambda c: c['protection'] in ('sc', 'none', 'milr'),
        lambda c: c['model'] in ('resnet50', ),
        lambda c: c['flips'] in ('rowhammer', )
    ),
    'milrmissing': (
        lambda c: c['dataset'] in ('imagenet_ds_128', 'driving_dataset_test'),
        lambda c: any((
            all((c['dataset'] == 'imagenet_ds_128', c['model'] != 'e2e')),
            all((c['dataset'] == 'driving_dataset_test', c['model'] == 'e2e'))
        )),
        lambda c: c['sampler'] == 'none',
        # ensure baseline execution
        lambda c: any((c['flips'] != 0, all((c['injection'] == 0, c['protection'] == 'none')))),
        # only baseline
        lambda c: isinstance(c['flips'], str) or isinstance(c['flips'], float) or c['flips'] == 0,
        lambda c: c['protection'] in ('sc', 'none', 'clipper', 'tmr', 'radar', 'milr', 'ranger'),
        lambda c: c['model'] not in ('vgg19',),
        lambda c: not c['quantization'],

        # retry
        lambda c: c['protection'] in ('milr', ),
        lambda c: c['model'] in ('resnet50', 'alexnet'),
        lambda c: isinstance(c['flips'], float),
    ),
    'quantized': (
        lambda c: c['dataset'] == 'imagenet_ds_128',
        lambda c: c['sampler'] == 'none',
        # ensure baseline execution
        lambda c: any((c['flips'] != 0, all((c['injection'] == 0, c['protection'] == 'none')))),
        # only baseline
        lambda c: c['protection'] in ('sc', 'none'),
        lambda c: c['flips'] in (0, 'row'),
        lambda c: c['model'] in ('alexnet', 'resnet50', 'googlenet'),
        lambda c: c['quantization']
    ),

    'optimal': (
        lambda c: c['dataset'] == 'imagenet_ds_128',
        lambda c: c['sampler'] == 'none',
        lambda c: c['model'] not in ('vgg19', 'e2e'),
        # ensure baseline execution
        lambda c: any((c['flips'] != 0, all((c['injection'] == 0, c['protection'] == 'none')))),
        lambda c: any((c['protection'] != 'none', all((c['injection'] == 0, c['flips'] == 0)))),
        lambda c: c['flips'] in PROBABILITIES + (0, ),
        # only baseline
        lambda c: c['protection'] in ('opt', ),
        lambda c: not c['quantization']
    ),
    'ecc': (
        lambda c: c['dataset'] == 'imagenet_ds_128',
        lambda c: c['sampler'] == 'none',
        lambda c: c['model'] not in ('vgg19', 'e2e'),
        # ensure baseline execution
        lambda c: any((c['flips'] != 0, all((c['injection'] == 0, c['protection'] == 'none')))),
        lambda c: any((c['protection'] != 'none', all((c['injection'] == 0, c['flips'] == 0)))),
        lambda c: c['flips'] in PROBABILITIES + (0, ),
        # only baseline
        lambda c: c['protection'] in ('secded', 'chipkill'),
        lambda c: not c['quantization']
    ),
    'ad': artifact_description_base + (lambda c: c['sampler'] == 'none',),
    'tinyad': artifact_description_base + (lambda c: c['sampler'] == 'tiny',),
    'resnet50coverage': artifact_description_base + (
        lambda c: c['sampler'] == 'none',
        lambda c: c['model'] == 'resnet50',
        lambda c: (c['flips'], c['protection']) in ((0, 'none'), (PROBABILITIES[0], 'opt')),
    ),
}

CONSTRAINTS = SHARDS_CONSTRAINTS[SHARD]


class Comparator:

    def __init__(self, k, v):
        self.k = k
        self.v = v

    def __call__(self, c):
        return c[self.k] == self.v


for overridden in eval(os.environ.get("CONSTRAINTS", "{}")).items():
    key, value = overridden
    CONSTRAINTS = CONSTRAINTS + (Comparator(key, value), )


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

if not SLURM_ARRAY:
    print("Ask for help from the authors, or stick to the sample commands!")
    exit(1)

INTERNAL_SLURM_ARRAY_TASK_ID = int(os.environ.get('INTERNAL_SLURM_ARRAY_TASK_ID', '0'))
CONFIG = SLURM_ARRAY[INTERNAL_SLURM_ARRAY_TASK_ID]

if __name__ == '__main__':
    file_names = set(get_storage_filename(i, {**DEFAULTS, 'injection': i['injection']}) for i in SLURM_ARRAY)
    for file_name in file_names:
        if os.path.exists(file_name):
            print(file_name)

    print('configs:', len(SLURM_ARRAY), 'jobs:', len(SLURM_ARRAY) / 40, 'files:', len(file_names))
