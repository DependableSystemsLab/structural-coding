# Structural Coding

Experimental code for Supercomputing 2022 submission 319

## Automatic Protection

Applying structural coding on a typical CNN with near zero effort:

```python
# linearcode/autonomy.py
from torchvision.models import resnet50
from linearcode.protection import apply_sc_automatically

model = resnet50(pretrained=True)
print(model)

model = apply_sc_automatically(model, n=256, k=32)
print(model)
```

## Running Demo via Docker
Running the `resnet50` network without protection with no fault:
```commandline
$ docker run --env CONSTRAINTS="{'dataset': 'imagenet', 'model': 'resnet50', 'sampler': 'tiny', 'flips': 0, 'protection': 'none'}" --env PRINT_STAT=1 dsn2022paper165/sc python map.py 
Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
100.0%
Done with batch 0 after injection
{'injection': 0, 'model': 'resnet50', 'quantization': False, 'sampler': 'tiny', 'dataset': 'imagenet', 'flips': 0, 'protection': 'none'}
accuracy 0.75 correct 12 all 16
loss 0.060732901096343994
```


Running the `resnet50` network without protection with `row` fault model:
```commandline
$ docker run --env CONSTRAINTS="{'dataset': 'imagenet', 'model': 'resnet50', 'sampler': 'tiny', 'flips': 'row', 'protection': 'none'}" --env PRINT_STAT=1 dsn2022paper165/sc python map.py 
Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
100.0%
Done with batch 0 after injection
{'injection': 0, 'model': 'resnet50', 'quantization': False, 'sampler': 'tiny', 'dataset': 'imagenet', 'flips': 'row', 'protection': 'none'}
Injecting 1193 faults at granularity 16
accuracy 0.0 correct 0 all 16
loss 4708924.5
```
Running the `resnet50` network with `sc` (Structural Coding) protection with `row` fault model:

```commandline
$ docker run --env CONSTRAINTS="{'dataset': 'imagenet', 'model': 'resnet50', 'sampler': 'tiny', 'flips': 'row', 'protection': 'sc'}" --env PRINT_STAT=1 dsn2022paper165/sc python map.py 
Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
100.0%
Done with batch 0 after injection
{'injection': 0, 'model': 'resnet50', 'quantization': False, 'sampler': 'tiny', 'dataset': 'imagenet', 'flips': 'row', 'protection': 'sc'}
Injecting 1193 faults at granularity 16
accuracy 0.75 correct 12 all 16
loss 0.061373598873615265
```

Running the `resnet50` network with `milr` (MILR) protection with `row` fault model:

```commandline
$ docker run --env CONSTRAINTS="{'dataset': 'imagenet', 'model': 'resnet50', 'sampler': 'tiny', 'flips': 'row', 'protection': 'milr'}" --env PRINT_STAT=1 dsn2022paper165/sc python map.py 
Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
100.0%[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.

Done with batch 0 after injection
{'injection': 0, 'model': 'resnet50', 'quantization': False, 'sampler': 'tiny', 'dataset': 'imagenet', 'flips': 'row', 'protection': 'milr'}
Injecting 1193 faults at granularity 16
accuracy 0.75 correct 12 all 16
loss 0.06021653115749359

```

## Run using Python virtual environment
[Assuming ubuntu linux]

[Create a virtual environment](https://docs.python.org/3/library/venv.html) and activate it:

```commandline
python3 -m venv venv
source venv/bin/activate
```

Within the root directory of the code, install the project requirements:
```
pip install -r requirements.txt
```

Set the PYTHONPATH environment variable:
```
export PYTHONPATH=`pwd`
```

Navigate to the experiment subdirectory:
```
cd linearcode
```

You can run the example experiment commands:
```
$ export CONSTRAINTS="{'dataset': 'imagenet', 'model': 'resnet50', 'sampler': 'tiny', 'flips': 'row', 'protection': 'milr'}"; export PRINT_STAT=1; python map.py
{'injection': 0, 'model': 'resnet50', 'quantization': False, 'sampler': 'tiny', 'dataset': 'imagenet', 'flips': 'row', 'protection': 'milr'}
[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
Injecting 1193 faults at granularity 16
Done with batch 0 after injection
accuracy 0.75 correct 12 all 16
loss 0.06021653115749359
```
