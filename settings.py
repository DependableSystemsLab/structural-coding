import os

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '16'))
INJECTIONS_RANGE = [int(i) for i in os.environ.get('INJECTIONS_RANGE', '0-400-1').split('-')]
IMAGENET1K_PATH = os.environ.get('IMAGENET1K_PATH', '../data/random20classes_FI')
IMAGENET_ROOT = os.environ.get('IMAGENET_ROOT', '../data/imagenet/')
IMAGENET_PATH = os.environ.get('IMAGENET_PATH', os.path.join(IMAGENET_ROOT, 'ILSVRC/Data/CLS-LOC'))
IMAGENET20P_SAMPLER_PATH = os.environ.get('IMAGENET20P_SAMPLER_PATH', '../imagenet20sampler.pkl')
COMMA_MODEL_ROOT = os.environ.get('COMMA_MODEL_ROOT', 'comma')
SHARD = os.environ.get('SHARD', 'default')
PROBABILITIES = (
    0.00000552972,
    # 0.00000552972 * 0.5 ** 2,
    0.00000552972 * 0.5 ** 4,
    # 0.00000552972 * 0.5 ** 6,
    0.00000552972 * 0.5 ** 8,
)

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
