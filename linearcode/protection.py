

def apply_sc(model):
    assert False


def apply_clipper(model):
    assert False


PROTECTIONS = {
    'none': lambda x: x,
    'sc': apply_sc,
    'clipper': apply_clipper,
}
