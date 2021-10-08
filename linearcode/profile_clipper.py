import sys

import torch
import torchvision.models

import common.models
from datasets import get_image_net_20p
from injection import ClipperReLU, convert, ClipperHardswish

model_classes = (
    # ('alexnet', torchvision.models.alexnet),
    ('squeezenet', torchvision.models.squeezenet1_1),
    # big memory requirement
    ('vgg19', torchvision.models.vgg19),
    ('mobilenet', torchvision.models.mobilenet_v3_small),
    ('googlenet', common.models.googlenet),
    # big memory requirement
    ('resnet50', common.models.resnet50),
    # ('shufflenet', torchvision.models.shufflenet_v2_x0_5),
)


if __name__ == '__main__':

    _, model, device = sys.argv

    for model_name, model_class in model_classes:
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
