import os
import pickle

import torch.nn

import injection
from complement.models import FashionMNISTTutorial
from datasets import get_fashion_mnist
from injection import convert

train = False

checkpoint_file_path = 'fashion_mnist_tutorial_smooth.pkl'
model = FashionMNISTTutorial(pretrained=True, weights=checkpoint_file_path)

if train:
    model, _ = convert(model, mapping={
        torch.nn.ReLU: injection.SmootherReLU
    })
else:
    model, _ = convert(model, mapping={
        torch.nn.ReLU: injection.ClipperReLU
    })
    for module in model.modules():
        if isinstance(module, injection.ClipperReLU):
            module.profile = True

training_data_loader = get_fashion_mnist(train=True)
evaluation_data_loader = get_fashion_mnist(train=False)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.003)
loss = torch.nn.CrossEntropyLoss()

if train:
    model.train()
else:
    model.eval()

for epoch in range(60):
    total_l = 0
    acc = []
    if train:
        model.zero_grad()
        if os.path.exists(checkpoint_file_path):
            with open(checkpoint_file_path, mode='rb') as checkpoint_file:
                model.load_state_dict(pickle.load(checkpoint_file))

    for x, y in training_data_loader:
        model_output = model(x)
        if train:
            l = loss(model_output, y)
            l.backward()
            total_l += float(l)
        acc.append(float(torch.sum(torch.topk(model_output, k=1).indices.flatten() == y) / len(y)))
    if train:
        optimizer.step()
        with open(checkpoint_file_path, mode='wb') as checkpoint_file:
            pickle.dump(model.state_dict(), checkpoint_file)
    print(total_l, sum(acc) / len(acc))
    if not train:
        for module in model.modules():
            if isinstance(module, injection.ClipperReLU):
                print(module.bounds)
        break





