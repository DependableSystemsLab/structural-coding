import torchvision.models
import time

from torch import Tensor

from settings import BATCH_SIZE
from datasets import get_image_net
import torch
import torch.nn.functional as F


from injection import convert

model = torchvision.models.vgg19(pretrained=True)
device = 'cuda:0'
model = model.to(device)
model.eval()


class BFTLinear(torch.nn.Linear):

    groups = None

    def forward(self, input: Tensor) -> Tensor:
        if self.out_features != 1000:
            return super().forward(input)
        outputs, features = self.weight.shape
        group_size = features // self.groups
        overlap_factor = 64
        perm = torch.randperm(features, device=device)
        results = []
        for start_index in range(0, features, group_size // overlap_factor):
            end_index = start_index + group_size
            rotated_index = torch.cat((perm[start_index: end_index], perm[:end_index % features]))
            results.append(torch.topk(F.linear(input[:, rotated_index],
                                      self.weight[:, rotated_index],
                                      self.bias / self.groups), k=1).indices)
        return torch.stack(results)

    @classmethod
    def from_original(cls, module: torch.nn.Linear):
        result = cls(in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None)
        result.weight = module.weight
        result.bias = module.bias
        return result


convert(model, {torch.nn.Linear: BFTLinear}, True)

imagenet = get_image_net()

for g in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    if g != 2:
        continue
    BFTLinear.groups = g
    start = time.time()
    correct = 0
    for x, y in imagenet:
        x = x.to(device)
        y = y.to(device)
        correct += int(torch.sum(torch.squeeze(torch.mode(model(x), dim=0).values) == torch.squeeze(y)))

    end = time.time()

    print(end - start, g, correct / 1000)

