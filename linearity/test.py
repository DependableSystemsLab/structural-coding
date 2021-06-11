import torch
from matplotlib import pyplot as plt
import torchvision

from datasets import get_data_loader

original_resnet = torchvision.models.vgg16(pretrained=True)


class Model(torch.nn.Module):
    poop = 0
    poop_index = 0

    def forward(self, x):
        x = original_resnet.features(x)

        x = original_resnet.avgpool(x)
        x = torch.flatten(x, 1)
        perturbation = torch.zeros(x.shape, device=x.device)
        perturbation[0][self.poop_index] = self.poop
        x = x + perturbation
        x = original_resnet.classifier(x)
        return x


data_loader = get_data_loader()

model = Model()

original_resnet.eval()

for g, p in enumerate(range(0, 900, 100)):
    model.poop_index = p
    _x = []
    _y = []
    for x, y in data_loader:
        for i in range(0, 2000, 200):
            Model.poop = i
            _x.append(Model.poop)
            model_output = model(x)
            first = float(model_output[0][122])
            top2 = torch.topk(model_output, k=2).values[0]
            top2_indices = torch.topk(model_output, k=2).indices[0]
            if 122 in top2_indices:
                second = float(top2[1])
            else:
                second = float(top2[0])
            _y.append((first, second))
            print(i)
        break
    plt.subplot(3, 3, g + 1)
    plt.title('Top 2 logits')
    plt.xlabel('deviation at fault point')
    plt.ylabel('propagated deviation at logits')
    plt.plot(_x, [__y[0] for __y in _y], marker='', label='target')
    plt.plot(_x, [__y[1] for __y in _y], marker='', label='second')
plt.tight_layout()
plt.legend()
plt.show()
