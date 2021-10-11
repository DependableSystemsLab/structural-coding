import numpy

from common.models import MODEL_CLASSES


if __name__ == '__main__':

    for model_name, model_class in MODEL_CLASSES:
        model = model_class()

        parameters = list(model.parameters())
        total_size = 0
        for i, p in enumerate(parameters):
            if len(p.shape) != 4:
                continue
            s = 1
            for d in p.shape:
                s *= d
            total_size += s
            # print(model_name, (s * 8) / (8 * 1024 * 8), p.shape)
        print(model_name, (total_size * 32) * 0.00000552972)
