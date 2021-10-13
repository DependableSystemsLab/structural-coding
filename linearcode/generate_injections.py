import numpy
import torch.nn

from common.models import MODEL_CLASSES


if __name__ == '__main__':

    for model_name, model_class in MODEL_CLASSES:
        model = model_class()
        b = 0
        w = 0
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                w += m.weight.flatten().shape[0]
        b = sum(p.flatten().shape[0] for p in model.parameters())
        b -= w
        print(model_name, '&', round(w * 100/ (w+ b), 2),'&', round(b * 100/ (w+ b), 2), end=' \\\\\n')
        continue
        parameters = list(model.parameters())
        total_size = 0
        for i, p in enumerate(parameters):
            if len(p.shape) != 4:
                continue
            s = 1
            for d in p.shape:
                s *= d
            total_size += s
            # print(model_name, (s * 8) / (8 * 1024 * 8), p.shape)
        print(model_name, (total_size * 32) * 0.00000552972)
