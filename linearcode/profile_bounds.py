import sys

import torch
import common.models
from datasets import get_image_net_20p
from injection import ClipperReLU, convert, ClipperHardswish


if __name__ == '__main__':

    _, model, device = sys.argv

    for model_name, model_class in common.models.MODEL_CLASSES:
        if model_name != model:
            continue
        model = model_class(pretrained=True)
        model.eval()
        model, max_injection_index = convert(model, mapping={
            torch.nn.ReLU: ClipperReLU,
            torch.nn.Hardswish: ClipperHardswish,
        })
        for m in model.modules():
            if isinstance(m, ClipperReLU) or isinstance(m, ClipperHardswish):
                m.train()
                m.profile = True
        model.to(device)
        percentage = 16015 // 100
        for i, (x, y) in enumerate(get_image_net_20p()):
            x = x.to(device)
            model(x)
            if i % percentage == 0:
                with open(model_name + '.txt', mode='w') as output_file:
                    for m in model.modules():
                        if hasattr(m, 'bounds'):
                            print(m.bounds, file=output_file)
                print(i // percentage, '%')
