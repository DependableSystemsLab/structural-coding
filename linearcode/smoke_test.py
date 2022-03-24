from linearcode.map import evaluate_config
from linearcode.parameters import SLURM_ARRAY

for config in SLURM_ARRAY:
    config['dataset'] = 'imagenet'
    config['sampler'] = 'tiny'
    if config['protection'] != 'opt':
        continue
    if config['injection'] >= 2:
        break
    print(config)
    evaluate_config(config)
