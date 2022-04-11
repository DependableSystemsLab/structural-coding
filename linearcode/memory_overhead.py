from common.models import MODEL_CLASSES
from linearcode.protection import PROTECTIONS
from storage import get_storage_filename

memory_filename = get_storage_filename({'fig': 'memory_overhead', 'protection': 'sc'},
                                       extension='.tex', storage='../thesis/data/')

with open(memory_filename, mode='w') as memory_file:
    for model_name, model_class in MODEL_CLASSES:
        if model_name in ('e2e', 'vgg19'):
            continue
        model = model_class()
        normalized_model = PROTECTIONS['before_quantization']['sc'](model, None)
        sc_correction_model = PROTECTIONS['after_quantization']['sc'](normalized_model, {'flips': 32})
        correction_size = sum(p.nelement() for p in sc_correction_model.parameters())
        model_size = sum(p.nelement() for p in model.parameters())
        print(model_name, round(100 * (correction_size / model_size - 1), 2), file=memory_file)

memory_filename = get_storage_filename({'fig': 'memory_overhead', 'protection': 'milr'},
                                       extension='.tex', storage='../thesis/data/')

with open(memory_filename, mode='w') as memory_file:
    for model_name, model_class in MODEL_CLASSES:
        if model_name in ('e2e', 'vgg19'):
            continue
        model = model_class()
        normalized_model = PROTECTIONS['before_quantization']['milr'](model, None)
        sc_correction_model = PROTECTIONS['after_quantization']['milr'](normalized_model, {'flips': 32})
        model_size = sum(p.nelement() for p in model.parameters())
        correction_size = sum(
            m.checkpoint.nelement() for m in sc_correction_model.modules() if hasattr(m, 'checkpoint')) + model_size
        print(model_name, round(100 * (correction_size / model_size - 1), 2), file=memory_file)

memory_filename = get_storage_filename({'fig': 'memory_overhead', 'protection': 'radar'},
                                       extension='.tex', storage='../thesis/data/')
with open(memory_filename, mode='w') as memory_file:
    for model_name, model_class in MODEL_CLASSES:
        if model_name in ('e2e', 'vgg19'):
            continue
        print(model_name, 25, file=memory_file)
