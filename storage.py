import os
import pickle


def get_storage_filename(key, defaults=None, extension='', storage='results'):
    if defaults is None:
        defaults = {}
    return os.path.join(storage, '-'.join(
        '{}:{}'.format(k, v) for k, v in sorted(key.items()) if defaults.get(k) != v
    ) + extension)


def store(key, value, defaults=None):
    filename = get_storage_filename(key, defaults)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename + '.tmp', mode='wb') as f:
        pickle.dump(value, f)
    os.rename(filename + '.tmp', filename + '.pkl')


def load(key, defaults=None, storage=None):
    try:
        filename = get_storage_filename(key, defaults, storage=storage)
        with open(filename + '.pkl', mode='rb') as f:
            return pickle.load(f)
    except OSError:
        return None


def extend(key, value, defaults):
    result = load(key, defaults) or []
    result.extend(value)
    store(key, value, defaults)

