import os
from itertools import product

DOMAIN = {
    'injection': range(400),
    'model': ('resnet50', 'alexnet', 'squeezenet', 'vgg19', 'mobilenet', 'googlenet', 'shufflenet', 'e2e'),
    'quantization': (True, False),
    'protection': ('none', 'clipper', 'ranger', 'sc', 'radar', 'milr', 'flr_mr', 'tmr'),
    'sampler': ('none', 'critical', 'tiny'),
    'dataset': ('full_imagenet', 'imagenet', 'imagenet_test',
                'imagenet_ds',
                # 10,000 deployment images as used in "Optimizing Selective Protection for CNN Resilience"
                'comma.ai', 'commaai_test'),
    'flips': (1, 2, 4, 8, 16, 32,
              0.00000552972,
              0.00000552972 * 0.5 ** 4,
              0.00000552972 * 0.5 ** 8,
              0.00000552972 * 0.5 ** 12,
              0.00000552972 * 0.5 ** 16,
              0)
}

# don't use short circuit execution here
CONSTRAINTS = (
    lambda c: c['sampler'] == 'none',
    lambda c: any((c['flips'] != 0, (c['injection'] == 0, c['protection'] == 'none'))),  # ensure baseline execution
    lambda c: c['protection'] in ('sc', 'none', 'clipper'),
    lambda c: c['model'] in ('resnet50', 'mobilenet'),
    lambda c: c['flips'] < 1,
    lambda c: not c['quantization'],
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

BASELINE_CONFIG = {k: DOMAIN[k][0] for k in DOMAIN}

SLURM_ARRAY = []


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


for constraint in CONSTRAINTS:
    constraint_keys = get_constraint_keys(constraint)
    if len(constraint_keys) == 1:
        domain = next(iter(constraint_keys))
        DOMAIN[domain] = tuple(v for v in DOMAIN[domain] if constraint({domain: v}))

for combination in product(*DOMAIN.values()):
    c = {k: v for k, v in zip(DOMAIN.keys(), combination)}
    if all(constraint(c) for constraint in CONSTRAINTS):
        SLURM_ARRAY.append(c)

CONFIG = SLURM_ARRAY[int(os.environ.get('INTERNAL_SLURM_ARRAY_TASK_ID'))]

if __name__ == '__main__':
    print(len(SLURM_ARRAY), len(SLURM_ARRAY) // 40)
