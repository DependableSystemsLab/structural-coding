# sudo sh -c 'echo -1 >/proc/sys/kernel/perf_event_paranoid'
# sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

from time import time

import torch
from common.models import MODEL_CLASSES
from linearcode.protection import PROTECTIONS
from storage import get_storage_filename

torch.random.manual_seed(0)
imagenet_image = torch.rand((16, 3, 299, 299))
e2e_image = torch.rand((1, 3, 200, 66))
n = 256


def measure_time(input_image, _model):
    count = 10
    start = time()
    for _ in range(count):
        _model.forward(input_image)
    return (time() - start) / count


for protection in (
        'sc',
        'milr',
        'radar',
):
    detection_filename = get_storage_filename({'fig': 'detection_time_overhead', 'protection': protection},
                                              extension='.tex', storage='../thesis/data/')

    with open(detection_filename, mode='w') as detection_file:
        for model_name, model_class in MODEL_CLASSES:
            if model_name in ('e2e', 'vgg19'):
                continue
            with torch.no_grad():

                if model_name == 'e2e':
                    image = e2e_image
                else:
                    image = imagenet_image
                now = time()
                model = model_class(pretrained=True)
                print(time() - now)
                model = model
                image = image

                baseline_flops = measure_time(image, model)
                sc_normalized_model = PROTECTIONS['before_quantization'][protection](model, None)
                sc_detection_model = PROTECTIONS['after_quantization'][protection](sc_normalized_model, {'flips': 1, 'n': 2048})
                sc_detection_flops = measure_time(image, sc_detection_model)

                print(model_name,
                      100 * (sc_detection_flops / baseline_flops - 1), file=detection_file)
                detection_file.flush()
                print()
