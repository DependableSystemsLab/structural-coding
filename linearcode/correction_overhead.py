# sudo sh -c 'echo -1 >/proc/sys/kernel/perf_event_paranoid'
# sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

from time import time

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
imagenet_image = torch.rand((16, 3, 299, 299))
e2e_image = torch.rand((1, 3, 200, 66))
n = 256


def count_flops(input_image, _model):
    papi_high.start_counters([papi_events.PAPI_DP_OPS])
    # papi_high.start_counters([papi_events.PAPI_SP_OPS])
    _model.forward(input_image)
    return papi_high.stop_counters()[0]


def measure_time(input_image, _model):
    count = 10
    start = time()
    for _ in range(count):
        _model.forward(input_image)
    return (time() - start) / count


measure = count_flops

memory_filename = get_storage_filename({'fig': 'memory_overhead'},
                                          extension='.tex', storage='../thesis/data/')


def corrupt_model(sc_correction_model, n, k):
    params = sorted(list(p for p in sc_correction_model.parameters() if len(p.shape) > 1),
                    key=lambda p: p.flatten().shape[0] / p.shape[0], reverse=True)
    p_iterator = iter(params)
    parameter = next(p_iterator)
    offset = 0

    with torch.no_grad():
        for i in range(k):
            while (i - offset) * (n + k) > parameter.shape[0]:
                parameter = next(p_iterator)
                offset = i
            parameter[(i - offset) * (n + k) % parameter.shape[0]] = 1e16


if __name__ == '__main__':
    with open(memory_filename, mode='w') as memory_file:
        for model_name, model_class in MODEL_CLASSES:
            model = model_class()
            sc_normalized_model = PROTECTIONS['before_quantization']['sc'](model, None)
            sc_correction_model = PROTECTIONS['after_quantization']['sc'](sc_normalized_model, {'flips': 32})
            correction_size = sum(p.nelement() for p in sc_correction_model.parameters())
            model_size = sum(p.nelement() for p in model.parameters())
            print(model_name, 100 * (correction_size / model_size - 1), file=memory_file)




    for protection in (
            'sc',
            # 'milr',
            # 'tmr'
    ):
        detection_filename = get_storage_filename({'fig': 'detection_flops_overhead', 'protection': protection},
                                                  extension='.tex', storage='../thesis/data/')

        with open(detection_filename, mode='w') as detection_file:
            for model_name, model_class in MODEL_CLASSES:
                # if model_name != 'shufflenet':
                #     continue
                correction_filename = get_storage_filename({'fig': 'correction_flops_overhead',
                                                            'model': model_name},
                                                           extension='.tex', storage='../thesis/data/')
                with open(correction_filename, mode='w') as correction_file:
                    with torch.no_grad():

                        if model_name == 'e2e':
                            image = e2e_image
                        else:
                            image = imagenet_image
                        now = time()
                        model = model_class(pretrained=True)
                        load_time = time() - now
                        model = model
                        image = image

                        now = time()
                        baseline_flops = measure(image, model)
                        inference_time = time() - now
                        sc_normalized_model = PROTECTIONS['before_quantization'][protection](model, None)
                        normalized_time = measure_time(image, sc_normalized_model)
                        sc_detection_model = PROTECTIONS['after_quantization'][protection](sc_normalized_model, {'flips': 1})
                        now = time()
                        sc_detection_flops = measure(image, sc_detection_model)
                        detection_time = time() - now

                        print(model_name,
                              100 * (sc_detection_flops / baseline_flops - 1), file=detection_file)
                        detection_file.flush()

                        for k in (1, 2, 4, 8):

                            sc_correction_model = PROTECTIONS['after_quantization']['sc'](sc_normalized_model, {'flips': 32})

                            # sc_normalization_flops = flops(image, sc_normalized_model)

                            corrupt_model(sc_correction_model, n, k)

                            now = time()
                            sc_correction_flops = measure(image, sc_correction_model)
                            correction_time = time() - now
                            print(model_name, load_time, inference_time, detection_time, normalized_time, correction_time)

                            print(k,
                                  100 * (sc_correction_flops / baseline_flops - 1), file=correction_file)
                            correction_file.flush()

                    print()
