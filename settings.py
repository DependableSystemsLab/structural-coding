import os

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '16'))
IMAGENET1K_PATH = os.environ.get('IMAGENET1K_PATH', '../data/random20classes_FI')
IMAGENET_ROOT = os.environ.get('IMAGENET_ROOT', '../data/imagenet/')
IMAGENET_PATH = os.environ.get('IMAGENET_PATH', os.path.join(IMAGENET_ROOT, 'ILSVRC/Data/CLS-LOC'))
IMAGENET20P_SAMPLER_PATH = os.environ.get('IMAGENET20P_SAMPLER_PATH', '../imagenet20sampler.pkl')
COMMA_MODEL_ROOT = os.environ.get('COMMA_MODEL_ROOT', 'comma')
