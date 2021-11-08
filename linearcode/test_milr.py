import torch.nn

from injection import MILRConv2d
from linearcode.protection import apply_milr

lm = torch.nn.Linear(512, 16)
cm = torch.nn.Conv2d(32, 128, (2, 2), stride=(2, 2))
m = torch.nn.Sequential(
    cm,
    torch.nn.Flatten(),
    lm,
)


image = torch.rand((1, 32, 4, 4))

correct_model_output = m(image)
m = apply_milr(m, None)
with torch.no_grad():
    cm.weight[0][0][0] = 1000
    corrupted_model_output = m(image)

print(torch.sum(correct_model_output - corrupted_model_output))
