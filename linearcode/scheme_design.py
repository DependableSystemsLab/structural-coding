from common.models import MODEL_CLASSES
from linearcode.fault import get_target_modules
from settings import PROBABILITIES
from scipy.stats import binom

from utils import biggest_divisor_smaller_than

print('BER', end=',')
for model_name, _ in MODEL_CLASSES:
    print(model_name, end=',')
print()

for protection_probability in (0.90, ):

    for ber in PROBABILITIES:
        print(ber, end=',')
        for model_name, model_class in MODEL_CLASSES:

            worst_scheme_percentage = 0
            worst_scheme = (0, 0)

            weighted_sum = 0
            sum_of_weights = 0

            model = model_class()

            for module in get_target_modules(model):
                channels = module.weight.shape[0]
                original_in_features = module.weight.shape[1]
                in_features = original_in_features
                in_features = biggest_divisor_smaller_than(in_features, 512)
                within_channel_bits = module.weight.nelement() // channels // (original_in_features // in_features) * module.weight.element_size() * 8
                channel_does_not_corrupt = (1 - ber) ** within_channel_bits
                channel_corrupts = 1 - channel_does_not_corrupt
                k = 1
                n = k + in_features
                while binom.cdf(k, n, channel_corrupts) < protection_probability:
                    k += 1
                    n += 1

                scheme_percentage = k / (n - k)
                weighted_sum = scheme_percentage * channels * within_channel_bits
                sum_of_weights = channels * within_channel_bits
                if scheme_percentage > worst_scheme_percentage:
                    worst_scheme_percentage = scheme_percentage
                    worst_scheme = (n, k)

                # print(model_name, ber, protection_probability, module.weight.shape, '({}, {})'.format(n, k))
            print(weighted_sum / sum_of_weights, end=',')
        print()