import torchvision.models

from pruning.parameters import CONFIG


def get_model():

    return getattr(torchvision.models, CONFIG['model'])(True)
