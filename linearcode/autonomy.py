from torchvision.models import resnet50
from linearcode.protection import apply_sc_automatically

model = resnet50(pretrained=True)
print(model)

model = apply_sc_automatically(model, n=256, k=32)
print(model)
