import os

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '16'))
IMAGENET1K_PATH = os.environ.get('IMAGENET1K_PATH', '../data/random20classes_FI')
IMAGENET_PATH = os.environ.get('IMAGENET_PATH', '../data/imagenet/ILSVRC/Data/CLS-LOC')
IMAGENET_ROOT = os.environ.get('IMAGENET_PATH', '../data/imagenet/')
IMAGENET20P_SAMPLER_PATH = os.environ.get('IMAGENET20P_SAMPLER_PATH', '../imagenet20sampler.pkl')
