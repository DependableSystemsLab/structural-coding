import torch
import torchvision.models
from torch import FloatTensor
from pypapi import papi_high
from pypapi import events as papi_events

from injection import convert, StructuralCodedConv2d, StructuralCodedLinear, ClipperReLU

torch.random.manual_seed(0)
image = torch.rand((1, 3, 299, 299))

print('k', end=',')

model_classes = (
    torchvision.models.vgg16,
    torchvision.models.resnet50,
    torchvision.models.alexnet,
)
for model_class in model_classes:
    print(model_class.__name__ + '-clipper',model_class.__name__ + '-sc-detection',model_class.__name__ + '-sc-correction', end=',', sep=',')
print()
for k in (1, 2, 4, 8, 16, 32):
    print(k, end=',')
    with torch.no_grad():
        for model_class in model_classes:
            model = model_class()
            papi_high.start_counters([papi_events.PAPI_SP_OPS])
            model.forward(image)
            baseline_flops = papi_high.stop_counters()[0]

            clipper_model, max_injection_index = convert(model, mapping={
                torch.nn.ReLU: ClipperReLU
            })

            relu_counter = 0
            for j, m in enumerate(clipper_model.modules()):
                if isinstance(m, torch.nn.ReLU):
                    m.bounds = 0, 50
                    m.module_index = relu_counter
                    relu_counter += 1

            papi_high.start_counters([papi_events.PAPI_SP_OPS])
            clipper_model.forward(image)
            clipper_flops = papi_high.stop_counters()[0]

            n = 256
            sc_model, _ = convert(model, mapping={
                torch.nn.Conv2d: StructuralCodedConv2d,
                torch.nn.Linear: StructuralCodedLinear,
            }, extra_kwargs={
                'k': k,
                'threshold': 1,
                'n': n
            })

            papi_high.start_counters([papi_events.PAPI_SP_OPS])
            sc_model.forward(image)
            sc_detection_flops = papi_high.stop_counters()[0]

            params = sorted(list(p for p in sc_model.parameters() if len(p.shape) > 2), key=lambda p: p.flatten().shape[0] / p.shape[0], reverse=True)

            for i in range(k):
                if len(params[0].shape) > 2:
                    params[0][i * (n + k) % params[0].shape[0]] = 1e16
                else:
                    params[0][:, i * (n + k) % params[0].shape[0]] = 1e16

            papi_high.start_counters([papi_events.PAPI_SP_OPS])
            sc_model.forward(image)
            sc_recovery_flops = papi_high.stop_counters()[0]

            print(100 * (clipper_flops - baseline_flops) / baseline_flops,
                  100 * (sc_detection_flops - baseline_flops) / baseline_flops,
                  100 * (sc_recovery_flops - baseline_flops) / baseline_flops, sep=',', end=',')
    print()