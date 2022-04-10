import math
import time
from pprint import pprint

from common.models import MODEL_CLASSES
from linearcode.fault import get_target_modules
from settings import PROBABILITIES
from scipy.stats import binom
from protection import normalize_model

print('BER', end=' ')
for ber in ('one-size-fits-all', ) + PROBABILITIES:
    print(ber, end=' ')
print()

optimal_points = {}

start_time = time.time()

for desired_protection_probability in (0.90, ):

    baseline_overhead = None

    for model_name, model_class in MODEL_CLASSES:
        if model_name == 'vgg19':
            continue

        inter_layer_binding = 1
        if model_name == 'resnet50':
            inter_layer_binding = 3

        desired_protection_probability = 1 - ((1 - desired_protection_probability) / inter_layer_binding)

        print(model_name, end=' ')
        for ber in ('one-size-fits-all', ) + PROBABILITIES:

            worst_scheme_percentage = 0
            worst_scheme = (0, 0)

            weighted_sum = 0
            sum_of_weights = 0

            model = model_class()
            model = normalize_model(model, None)

            for module in get_target_modules(model):
                channels = module.weight.shape[0]
                original_in_features = module.weight.shape[1]
                in_features = original_in_features
                within_channel_bits = module.weight.nelement() // channels // (original_in_features // in_features) * module.weight.element_size() * 8
                if ber != 'one-size-fits-all':
                    channel_does_not_corrupt = (1 - ber) ** within_channel_bits
                    channel_corrupts = 1 - channel_does_not_corrupt
                    k = 1
                    n = k + 256
                    protection_probability = binom.cdf(k, n, channel_corrupts)
                    while protection_probability < desired_protection_probability:
                        k += 1
                        n += 1
                        protection_probability = binom.cdf(k, n, channel_corrupts)
                else:
                    k = 32
                    n = k + 256

                optimal_points[tuple(module.weight.shape) + (ber, model_name)] = (n, k)
                scheme_percentage = k * math.ceil(channels / n) / channels
                weighted_sum += k * math.ceil(channels / n) * within_channel_bits
                sum_of_weights += channels * within_channel_bits
                if scheme_percentage > worst_scheme_percentage:
                    worst_scheme_percentage = scheme_percentage
                    worst_scheme = (n, k)

                # print(model_name, ber, protection_probability, module.weight.shape, '({}, {})'.format(n, k))
            overhead_percentage = weighted_sum / sum_of_weights
            if ber == 'one-size-fits-all':
                baseline_overhead = overhead_percentage
            print(100 * (baseline_overhead - overhead_percentage) / baseline_overhead, end=' ')
        print()

pprint(optimal_points)
print("Finding optimal points took ", time.time() - start_time)