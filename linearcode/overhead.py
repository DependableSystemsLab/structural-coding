# sudo sh -c 'echo -1 >/proc/sys/kernel/perf_event_paranoid'


import torch
import torchvision.models
from torch import FloatTensor
from pypapi import papi_high
from pypapi import events as papi_events

from common.models import MODEL_CLASSES
from injection import convert, StructuralCodedConv2d, StructuralCodedLinear, ClipperReLU

# for i in range(32, 257):
#     inp = torch.randn(16, 32, 229, 229)
#     w = torch.randn(i, 32, 4, 5)
#     papi_high.start_counters([papi_events.PAPI_FP_OPS])
#     torch.conv2d(inp, w)
#     print(i, papi_high.stop_counters()[0], sep=',')
# exit()
from linearcode.protection import PROTECTIONS

torch.random.manual_seed(0)
imagenet_image = torch.rand((1, 3, 299, 299))
e2e_image = torch.rand((1, 3, 200, 66))
n = 256


def flops(input_image, _model):
    overhead = 0
    for _ in range(5):
        input_image = torch.rand(input_image.shape)
        input_image = input_image.double()
        papi_high.start_counters([papi_events.PAPI_DP_OPS])
        # papi_high.start_counters([papi_events.PAPI_SP_OPS])
        _model.forward(input_image)
        overhead = max(overhead, papi_high.stop_counters()[0])
    return overhead


for k in (1, 2, 4, 8, 16, 32):
    with torch.no_grad():
        for model_name, model_class in MODEL_CLASSES:
            if model_name == 'e2e':
                image = e2e_image
            else:
                image = imagenet_image
            model = model_class()
            model = model.double()

            baseline_flops = flops(image, model)
            sc_normalized_model = PROTECTIONS['before_quantization']['sc'](model, None)
            sc_detection_model = PROTECTIONS['after_quantization']['sc'](sc_normalized_model, {'flips': 1})
            sc_correction_model = PROTECTIONS['after_quantization']['sc'](sc_detection_model, {'flips': k})

            sc_detection_flops = flops(image, sc_detection_model)
            sc_normalization_flops = flops(image, sc_normalized_model)

            params = sorted(list(p for p in sc_correction_model.parameters() if len(p.shape) > 2), key=lambda p: p.flatten().shape[0] / p.shape[0], reverse=True)

            for i in range(k):
                if len(params[0].shape) > 2:
                    params[0][i * (n + k) % params[0].shape[0]] = 1e16
                else:
                    params[0][:, i * (n + k) % params[0].shape[0]] = 1e16

            sc_correction_flops = flops(image, sc_correction_model)

            print(model_name,
                  # 100 * (sc_normalization_flops / baseline_flops - 1),
                  100 * (sc_detection_flops / sc_normalization_flops - 1),
                  100 * (sc_correction_flops / sc_normalization_flops - 1))

    print()