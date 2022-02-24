import torch

from common.models import MODEL_CLASSES
from datasets import get_image_net
from injection import convert
from linearcode.protection import PROTECTIONS
from storage import get_storage_filename

n = 256
k = 32


def get_accuracy(m):
    data_loader = get_image_net()
    total = 0
    correct = 0
    for x, y in data_loader:
        correct += int(torch.sum(torch.topk(m(x), 1).indices.squeeze() == y))
        total += len(x)
        print(total)
    return correct / total


for model_name, model_class in MODEL_CLASSES:

    if model_name in ('e2e', 'alexnet', 'vgg19', 'squeezenet'):
        continue

    correction_filename = get_storage_filename({'fig': 'stability',
                                                'model': model_name},
                                               extension='.tex', storage='../thesis/data/')
    with open(correction_filename, mode='w') as correction_file:
        model = model_class(pretrained=True)
        model.eval()

        baseline = get_accuracy(model)
        sc_correction_model = PROTECTIONS['after_quantization']['sc'](model, {'flips': k})
        params = sorted(list(p for p in sc_correction_model.parameters() if len(p.shape) > 1),
                        key=lambda p: p.flatten().shape[0] / p.shape[0], reverse=True)

        p_iterator = iter(params)
        parameter = next(p_iterator)
        offset = 0
        print(0, baseline, file=correction_file)
        for i in range(1, 11):
            while (i - offset) * (n + k) > parameter.shape[0]:
                parameter = next(p_iterator)
                offset = i
            with torch.no_grad():
                parameter[(i - offset) * (n + k) % parameter.shape[0]] = 1e16
                print(i, get_accuracy(sc_correction_model), file=correction_file)
                correction_file.flush()
