import torch.nn

from injection import MILRConv2d
from linearcode.protection import apply_milr

lm = torch.nn.Linear(1152, 16)
m = torch.nn.Sequential(
    torch.nn.Conv2d(32, 128, (2, 2)),
    torch.nn.Flatten(),
    lm,
)


image = torch.rand((1, 32, 4, 4))

correct_model_output = m(image)
m = apply_milr(m, None)
with torch.no_grad():
    lm.weight[0][0] = 1000
    corrupted_model_output = m(image)

print(torch.sum(correct_model_output - corrupted_model_output))
