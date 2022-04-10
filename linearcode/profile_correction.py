import time

from torchvision.models import resnet50

from linearcode.correction_overhead import corrupt_model
from linearcode.protection import apply_sc_automatically
import torch
import cProfile

n = 8
k = 1

imagenet_image = torch.rand((16, 3, 299, 299))


def correct():
    with torch.no_grad():
        now = time.time()
        model = resnet50(pretrained=True)
        load_time = time.time() - now

        now = time.time()
        model(imagenet_image)
        inference = time.time() - now

        model = apply_sc_automatically(model, n=n, k=k)

        torch.random.manual_seed(0)

        corrupt_model(model, n, 1)

        now = time.time()
        model(imagenet_image)
        correction_and_inference = time.time() - now

        print(load_time, inference, correction_and_inference - inference)


p = cProfile.Profile()
p.run('correct()')

p.dump_stats('correction.pstats')
