

#  create model
import os
import sys
import time

import torch
from numpy.distutils.fcompiler import str2bool

from common.models import MODEL_CLASSES, LOSS_CLASSES
from datasets import get_dataset
from linearcode.fault import inject_memory_fault
from linearcode.parameters import CONFIG, DEFAULTS
from linearcode.protection import PROTECTIONS
from settings import BATCH_SIZE
from storage import extend

model_class = dict(MODEL_CLASSES)[CONFIG['model']]
loss_class = dict(LOSS_CLASSES)[CONFIG['model']]
loss_function = loss_class()
model = model_class(pretrained=True)

#  protect model
model = PROTECTIONS['before_quantization'].get(CONFIG['protection'], lambda t, _: t)(model, CONFIG)

if CONFIG['quantization']:
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    quantization_state_filename = CONFIG['model']
    if CONFIG['protection'] == 'sc':
        quantization_state_filename = 'normalized_' + quantization_state_filename
    quantization_state_dict = torch.load('quants/' + quantization_state_filename + '.pth')
    model.load_state_dict(quantization_state_dict, False)


#  protect model
model = PROTECTIONS['after_quantization'].get(CONFIG['protection'], lambda t, _: t)(model, CONFIG)

model.eval()
if CONFIG['quantization']:
    model.apply(torch.quantization.disable_observer)

# corrupt model
inject_memory_fault(model, CONFIG)


dataset = get_dataset(CONFIG)

# evaluate
with torch.no_grad():
    evaluation = []
    correct = 0
    total = 0
    for i, (x, y) in enumerate(dataset):
        protection_modules = []
        for m in model.modules():
            if hasattr(m, 'reset_internal_log'):
                m.reset_internal_log()
                protection_modules.append(m)
        start_time = time.time()
        model_output = model(x)
        loss = loss_function(model_output, y)
        if CONFIG['model'] == 'e2e':
            indices = None
        else:
            indices = torch.topk(model_output, k=5).indices
        end_time = time.time()
        evaluation.append({'top5': indices,
                           'label': y,
                           'batch': i,
                           'config': CONFIG,
                           'loss': loss,
                           'elapsed_time': end_time - start_time,
                           'protection': [m.get_internal_log() for m in protection_modules],
                           'batch_size': BATCH_SIZE})
        print("Done with batch {} after injection".format(i), file=sys.stderr)
        correct += int(torch.sum(indices[:, 0] == y))
        total += len(x)
    if str2bool(os.environ.get('PRINT_STAT', '0')):
        print('accuracy', float(correct / total), 'correct', correct, 'all', total)
        total_loss = 0
        total = 0
        for e in evaluation:
            total_loss += torch.sum(e['loss'])
            total += len(e['label'])
        print('loss', float(total_loss / total))
    extend(CONFIG, evaluation, {**DEFAULTS, 'injection': CONFIG['injection']})
