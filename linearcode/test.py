import torch

from datasets import get_image_net
from injection import bitflip, convert, ReorderingCodedConv2d
from linearcode.models import get_model

model = get_model()
model.eval()

dataset = get_image_net(sampler=(4, 10, 14, 16, 23, 27, 39, 51, 53, 64, 68, 109, 111, 120, 124, 131, 139,
                                 143, 162, 215, 236, 242, 276, 284, 303, 332, 374, 384, 397, 405, 408, 413,
                                 419, 420, 423, 424, 431, 432, 447, 448, 462, 466, 485, 502, 503, 511, 532,
                                 536, 538, 540, 563, 581, 621, 662, 673, 677, 690, 693, 701, 733, 767, 774,
                                 784, 789, 806, 808, 828, 851, 872, 877, 885, 907, 912, 915, 928, 929, 934,
                                 948, 966, 998))

convert(model, {torch.nn.Conv2d: ReorderingCodedConv2d}, in_place=True)

parameters = list(model.parameters())

for j, (x, y) in enumerate(dataset):
    for i in range(64):
        if (j, i) not in (
                (2, 2),
                (4, 4),
        ):
            continue
        tensor_index = (i, 0, 1, 2)
        layer_index = 6

        parameter = parameters[layer_index]
        target = None
        for m in model.modules():
            if isinstance(m, ReorderingCodedConv2d):
                if m.weight is parameter:
                    m.observe = True
                    target = m
        parameter_value = parameter[tensor_index]
        corrupted = bitflip(parameter_value, 25)
        print(parameter_value, '->', corrupted)
        with torch.no_grad():
            golden = torch.argmax(model(x)) == y[0]
            print(golden)
            parameter[tensor_index] = corrupted
            faulty = torch.argmax(model(x)) == y[0]
            print(faulty)
            if golden and not faulty:
                print(">>>>>>>>>>>>>> BAD DATAPOINT ", j, i)
            if golden != faulty and target.observe:
                print(">>>>>>>>>>>>>> VERY BAD DETECTION")
            parameter[tensor_index] = parameter_value


