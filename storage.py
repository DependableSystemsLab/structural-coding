import os
import pickle


def get_storage_filename(key, defaults=None, extension='', storage=None):
    if defaults is None:
        defaults = {}
    if storage is None:
        storage = 'results'
    return os.path.join(storage, '-'.join(
        '{}:{}'.format(k, v) for k, v in sorted(key.items()) if defaults.get(k) != v
    ) + extension)


def store(key, value, defaults=None, append=False):
    mode = 'wb'
    if append:
        mode = 'ab'
    filename = get_storage_filename(key, defaults)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode=mode) as f:
        pickle.dump(value, f)


def load(key, defaults=None, storage=None):
    filename = get_storage_filename(key, defaults, storage=storage)
    return load_pickle(filename)


def load_pickle(filename):
    try:
        result = []
        with open(filename, mode='rb') as f:
            try:
                while True:
                    result.append(pickle.load(f))
                    # return result
            except EOFError:
                return result
    except OSError as ex:
        print("Error reading", ex, filename)
        return None


def extend(key, value, defaults):
    store(key, value, defaults, True)


if __name__ == '__main__':
    print(len(load_pickle('linearcode/results/dataset:imagenet_ds_128-flips:2.160046875e-08-model:resnet50-protection:none')))