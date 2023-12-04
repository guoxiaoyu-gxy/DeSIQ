import copy
import logging

import numpy as np

from utils import constants
import os
os.environ['TRANSFORMERS_CACHE'] = constants.TRANSFORMERS_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = constants.DATASETS_CACHE_DIR

import numpy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from transformers import set_seed
from transformers import BertTokenizer, BertModel
from transformers import T5EncoderModel, T5Tokenizer

import random
import json
import pandas as pd

from utils.get_statistics import *
from socialiq2.text_only_model import T5EncoderTextOnlyModelDouble, JudgeText
from socialiq2.text_only_model import read_dataset, reform_dataset

set_seed(42)
logger = logging.getLogger(__name__)

# parameters
batch_size = 16
t5_model_selection = 't5-small'
t5_output_dim = 1024

bert_model_selection = 'bert-base-uncased'

separate = False
load_data = 'a'
if load_data == 'a':
    t5_output_dim = t5_output_dim
    separate = True
elif load_data == 'qa' or load_data == 'at':
    t5_output_dim *= 2
    separate = True
elif load_data == 'qat':
    t5_output_dim *= 3
    separate = True
elif load_data == 'qta' or load_data == 'qac':
    t5_output_dim = t5_output_dim
    separate = False

perturbation = False


def calc_accuracy(predictions, labels):
    predictions_ = predictions.cpu().detach().numpy()
    labels_ = labels.cpu().detach().numpy()
    index = np.argmax(predictions_, axis=1)
    return (numpy.array(index) == numpy.array(labels_)).sum() / predictions.shape[0]


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    directory = "siq2/qa"
    transcript_directory = "siq2/transcript"

    # dek = read_dataset(directory, transcript_directory, 'val')
    dek = reform_dataset(directory, transcript_directory, 'val', way='rawa')

    tokenizer = T5Tokenizer.from_pretrained(t5_model_selection)
    # model = T5EncoderTextOnlyModel().to(device)
    model = T5EncoderTextOnlyModelDouble().to(device)
    judge_model = JudgeText().to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(judge_model.parameters()), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    dev_acc = []
    dev_acc_decrease_iters = 0

    checkpoint = torch.load('best.pt')
    model.load_state_dict(checkpoint['modelA_state_dict'])
    judge_model.load_state_dict(checkpoint['modelB_state_dict'])

    _accs = []
    model.eval()
    judge_model.eval()
    ds_size = len(dek)
    if ds_size % batch_size == 0:
        iters = int(ds_size / batch_size)
    else:
        iters = int(ds_size / batch_size) + 1
    for j in range(iters):
        this_dek = dek[j * batch_size:(j + 1) * batch_size]

        q = tokenizer(this_dek['q'].values.tolist(), return_tensors='pt', max_length=50,
                      truncation=True, padding='max_length').input_ids.to(device)
        t = tokenizer(this_dek['t'].values.tolist(), return_tensors='pt', max_length=500,
                      truncation=True, padding='max_length').input_ids.to(device)
        if separate:
            a0 = tokenizer(this_dek['a0'].values.tolist(), return_tensors='pt', max_length=50,
                           truncation=True, padding='max_length').input_ids.to(device)
            a1 = tokenizer(this_dek['a1'].values.tolist(), return_tensors='pt', max_length=50,
                           truncation=True, padding='max_length').input_ids.to(device)
            a2 = tokenizer(this_dek['a2'].values.tolist(), return_tensors='pt', max_length=50,
                           truncation=True, padding='max_length').input_ids.to(device)
            a3 = tokenizer(this_dek['a3'].values.tolist(), return_tensors='pt', max_length=50,
                           truncation=True, padding='max_length').input_ids.to(device)
            q, a0, a1, a2, a3, t = model(q, a0, a1, a2, a3, t)
        elif load_data == 'qac':
            qa0 = tokenizer(this_dek['qa0'].values.tolist(), return_tensors='pt', max_length=500,
                            truncation=True, padding='max_length').input_ids.to(device)
            qa1 = tokenizer(this_dek['qa1'].values.tolist(), return_tensors='pt', max_length=500,
                            truncation=True, padding='max_length').input_ids.to(device)
            qa2 = tokenizer(this_dek['qa2'].values.tolist(), return_tensors='pt', max_length=500,
                            truncation=True, padding='max_length').input_ids.to(device)
            qa3 = tokenizer(this_dek['qa3'].values.tolist(), return_tensors='pt', max_length=500,
                            truncation=True, padding='max_length').input_ids.to(device)
            q, qa0, qa1, qa2, qa3, t = model(q, qa0, qa1, qa2, qa3, t)
        else:
            qta0 = tokenizer(this_dek['qta0'].values.tolist(), return_tensors='pt', max_length=500,
                             truncation=True, padding='max_length').input_ids.to(device)
            qta1 = tokenizer(this_dek['qta1'].values.tolist(), return_tensors='pt', max_length=500,
                             truncation=True, padding='max_length').input_ids.to(device)
            qta2 = tokenizer(this_dek['qta2'].values.tolist(), return_tensors='pt', max_length=500,
                             truncation=True, padding='max_length').input_ids.to(device)
            qta3 = tokenizer(this_dek['qta3'].values.tolist(), return_tensors='pt', max_length=500,
                             truncation=True, padding='max_length').input_ids.to(device)
            q, qta0, qta1, qta2, qta3, t = model(q, qta0, qta1, qta2, qta3, t)

        labels = torch.LongTensor(this_dek['a_idx'].values.tolist()).to(device)

        if load_data == 'a':
            a0 = judge_model(a0)
            a1 = judge_model(a1)
            a2 = judge_model(a2)
            a3 = judge_model(a3)
        elif load_data == 'qa':
            a0 = judge_model(torch.cat((q, a0), dim=1))
            a1 = judge_model(torch.cat((q, a1), dim=1))
            a2 = judge_model(torch.cat((q, a2), dim=1))
            a3 = judge_model(torch.cat((q, a3), dim=1))
        elif load_data == 'at':
            a0 = judge_model(torch.cat((t, a0), dim=1))
            a1 = judge_model(torch.cat((t, a1), dim=1))
            a2 = judge_model(torch.cat((t, a2), dim=1))
            a3 = judge_model(torch.cat((t, a3), dim=1))
        elif load_data == 'qat':
            a0 = judge_model(torch.cat((q, a0, t), dim=1))
            a1 = judge_model(torch.cat((q, a1, t), dim=1))
            a2 = judge_model(torch.cat((q, a2, t), dim=1))
            a3 = judge_model(torch.cat((q, a3, t), dim=1))
        elif load_data == 'qta':
            a0 = judge_model(qta0)
            a1 = judge_model(qta1)
            a2 = judge_model(qta2)
            a3 = judge_model(qta3)
        elif load_data == 'qac':
            a0 = judge_model(qa0)
            a1 = judge_model(qa1)
            a2 = judge_model(qa2)
            a3 = judge_model(qa3)
        a = torch.cat((a0, a1, a2, a3), dim=1)
        _accs.append(calc_accuracy(a, labels))

    print("Dev Accs %f", numpy.array(_accs, dtype="float32").mean())
    print("-----------")
