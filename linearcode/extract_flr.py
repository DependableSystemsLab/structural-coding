import pickle
from collections import defaultdict

import numpy

from common.models import MODEL_CLASSES
from linearcode.fault import get_flattened_weights
from storage import load, get_storage_filename

for model_name, model_class in MODEL_CLASSES:
    if model_name != 'squeezenet':
        continue

    baseline = get_storage_filename({'dataset': 'imagenet_as_i',
                                     'flips': 0,
                                     'model': model_name,
                                     'protection': 'none'})

    injection = get_storage_filename({'dataset': 'imagenet_as_i',
                                      'flips': 'flr',
                                      'model': model_name,
                                      'protection': 'none'})

    if model_name in ('e2e', 'vgg19'):
        continue

    model = model_class()

    parameters = get_flattened_weights(model)
    modules, parameters = zip(*parameters)

    baseline_losses = {}
    injection_losses = {}
    counter = 0
    with open(baseline, mode='rb') as f:
        while True:
            try:
                b = pickle.load(f)
                injection_ = b[0]['config']['injection']
                if injection_ in baseline_losses:
                    counter += 1
                baseline_losses[injection_] = float(b[0]['loss'])
            except EOFError:
                break
    print(counter, len(baseline_losses))
    # exit()
    with open(injection, mode='rb') as f:
        while True:
            try:
                i = pickle.load(f)
                injection_losses[i[0]['config']['injection']] = float(i[0]['loss'])
            except EOFError:
                break

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
