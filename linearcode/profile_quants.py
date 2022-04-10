import sys

import torch
import common.models
from datasets import get_image_net
from injection import ClipperReLU, convert, ClipperHardswish
from linearcode.protection import apply_sc

normalized = True


if __name__ == '__main__':

    _, model, device = sys.argv

    for model_name, model_class in common.models.MODEL_CLASSES:
        # if model_name != model:
        #     continue
        model = model_class(pretrained=True)
        if normalized:
            model = apply_sc(model, None)
            model_name = 'normalized_' + model_name
        initial_states = model.state_dict().keys()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)
        model.eval()
        percentage = 16015 // 100
        percentage = 1
        for i, (x, y) in enumerate(get_image_net()):
            x = x.to(device)
            model(x)
            if i % percentage == 0:
                filename = 'quants/' + model_name + '.pth'
                torch.save({k: v for k, v in model.state_dict().items() if k not in initial_states}, filename)
                print(i // percentage, '%')
                break
