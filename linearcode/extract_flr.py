from collections import defaultdict

import numpy

from common.models import MODEL_CLASSES
from linearcode.fault import get_flattened_weights
from storage import load

for model_name, model_class in MODEL_CLASSES:

    baseline = load({'dataset': 'imagenet_as_i',
                     'flips': 0,
                     'model': 'resnet50',
                     'protection': 'none'})

    injection = load({'dataset': 'imagenet_as_i',
                      'flips': 'flr',
                      'model': 'resnet50',
                      'protection': 'none'})

    if model_name in ('e2e', 'vgg19'):
        continue

    model = model_class()

    parameters = get_flattened_weights(model)
    modules, parameters = zip(*parameters)

    baseline_losses = {}
    injection_losses = {}
    counter = 0
    for b in baseline:
        injection_ = b[0]['config']['injection']
        if injection_ in baseline_losses:
            counter += 1
        baseline_losses[injection_] = b[0]['loss']
    print(counter, len(baseline_losses))
    # exit()
    for i in injection:
        injection_losses[i[0]['config']['injection']] = i[0]['loss']

    common_injections = set(baseline_losses.keys()) & set(injection_losses.keys())
    delta_loss = defaultdict(float)
    delta_loss_count = defaultdict(int)

    fmap_count = sum(m.weight.shape[0] for m in modules)

    for i in common_injections:
        fmap_index = i % fmap_count

        difference = injection_losses[i] - baseline_losses[i]
        if not numpy.isnan(difference) and not numpy.isinf(difference):
            delta_loss[fmap_index] += float(abs(difference))
            delta_loss_count[fmap_index] += 1

    with open('flr/{}.txt'.format(model_name), mode='w') as output_file:
        for i in delta_loss:
            print(i, delta_loss[i] / delta_loss_count[i], delta_loss_count[i], file=output_file)

