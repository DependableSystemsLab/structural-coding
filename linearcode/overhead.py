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
from storage import get_storage_filename

torch.random.manual_seed(0)
imagenet_image = torch.rand((1, 3, 299, 299))
e2e_image = torch.rand((1, 3, 200, 66))
n = 256


def flops(input_image, _model):
    papi_high.start_counters([papi_events.PAPI_DP_OPS])
    # papi_high.start_counters([papi_events.PAPI_SP_OPS])
    _model.forward(input_image)
    return papi_high.stop_counters()[0]


for model_name, model_class in MODEL_CLASSES:
    if model_name != 'mobilenet':
        continue
    correction_filename = get_storage_filename({'fig': 'correction_flops_overhead',
                                                'model': model_name},
                                               extension='.tex', storage='../ubcthesis/data/')

    detection_filename = get_storage_filename({'fig': 'detection_flops_overhead'},
                                              extension='.tex', storage='../ubcthesis/data/')
    with open(detection_filename, mode='w') as detection_file:
        with open(correction_filename, mode='w') as correction_file:
            with torch.no_grad():

                if model_name == 'e2e':
                    image = e2e_image
                else:
                    image = imagenet_image
                model = model_class()
                model = model.double()
                image = image.double()

                baseline_flops = flops(image, model)
                sc_normalized_model = PROTECTIONS['before_quantization']['sc'](model, None)
                sc_detection_model = PROTECTIONS['after_quantization']['sc'](sc_normalized_model, {'flips': 1})
                sc_detection_flops = flops(image, sc_detection_model)

                print(model_name,
                      100 * (sc_detection_flops / baseline_flops - 1), file=detection_file)
                detection_file.flush()

                for k in (1, 2, 4, 8, 16, 32):

                    sc_correction_model = PROTECTIONS['after_quantization']['sc'](sc_normalized_model, {'flips': 32})

                    # sc_normalization_flops = flops(image, sc_normalized_model)

                    params = sorted(list(p for p in sc_correction_model.parameters() if len(p.shape) > 1),
                                    key=lambda p: p.flatten().shape[0] / p.shape[0], reverse=True)

                    p_iterator = iter(params)
                    parameter = next(p_iterator)
                    offset = 0
                    for i in range(k):
                        while (i - offset) * (n + k) > parameter.shape[0]:
                            parameter = next(p_iterator)
                            offset = i
                        parameter[(i - offset) * (n + k) % parameter.shape[0]] = 1e16

                    sc_correction_flops = flops(image, sc_correction_model)

                    print(k,
                          100 * (sc_correction_flops / baseline_flops - 1), file=correction_file)
                    correction_file.flush()

            print()
