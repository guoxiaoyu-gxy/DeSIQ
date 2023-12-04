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


set_seed(42)
logger = logging.getLogger(__name__)

# parameters
batch_size = 16
t5_model_selection = 't5-small'
t5_output_dim = 1024

bert_model_selection = 'bert-base-uncased'

separate = False
load_data = 'qat'
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


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.text_encoder = BertModel.from_pretrained(bert_model_selection)

    def forward(self, x):
        x = self.text_encoder(x).last_hidden_state
        return x


class T5EncoderTextOnlyModel(nn.Module):
    def __init__(self):
        super(T5EncoderTextOnlyModel, self).__init__()
        self.t5_model = T5EncoderModel.from_pretrained(t5_model_selection)

    def forward(self, q, a0, a1, a2, a3, t):
        q = self.t5_model(q).last_hidden_state[:, -1, :]
        a0 = self.t5_model(a0).last_hidden_state[:, -1, :]
        a1 = self.t5_model(a1).last_hidden_state[:, -1, :]
        a2 = self.t5_model(a2).last_hidden_state[:, -1, :]
        a3 = self.t5_model(a3).last_hidden_state[:, -1, :]
        t = self.t5_model(t).last_hidden_state[:, -1, :]
        return q, a0, a1, a2, a3, t


class T5EncoderTextOnlyModelDouble(nn.Module):
    def __init__(self):
        super(T5EncoderTextOnlyModelDouble, self).__init__()
        self.t5_model = T5EncoderModel.from_pretrained(t5_model_selection)

    def forward(self, q, a0, a1, a2, a3, t):
        q = self.t5_model(q).last_hidden_state
        a0 = self.t5_model(a0).last_hidden_state
        a1 = self.t5_model(a1).last_hidden_state
        a2 = self.t5_model(a2).last_hidden_state
        a3 = self.t5_model(a3).last_hidden_state
        t = self.t5_model(t).last_hidden_state
        q = torch.cat((q[:, 0, :], q[:, -1, :]), dim=-1)
        a0 = torch.cat((a0[:, 0, :], a0[:, -1, :]), dim=-1)
        a1 = torch.cat((a1[:, 0, :], a1[:, -1, :]), dim=-1)
        a2 = torch.cat((a2[:, 0, :], a2[:, -1, :]), dim=-1)
        a3 = torch.cat((a3[:, 0, :], a3[:, -1, :]), dim=-1)
        t = torch.cat((t[:, 0, :], t[:, -1, :]), dim=-1)
        return q, a0, a1, a2, a3, t


class JudgeText(nn.Module):
    def __init__(self):
        super(JudgeText, self).__init__()
        self.linear1 = nn.Linear(t5_output_dim, 25)
        self.activation = torch.nn.Sigmoid()
        self.linear2 = nn.Linear(25, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        return x


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def read_dataset(directory, transcript_directory, split):
    data = []
    transcripts = get_transcripts_from_directory(transcript_directory)
    with open(os.path.join(directory, 'qa_' + split + '.json'), 'r') as f:
        json_data = [json.loads(line) for line in f]
    for item in json_data:
        vid = item['qid'].split('_')[0]
        if vid in transcripts:
            transcript = ' '.join(transcripts[vid])
            data.append([item['q'], item['a0'], item['a1'], item['a2'], item['a3'], item['answer_idx'], transcript])
    df = pd.DataFrame(data, columns=['q', 'a0', 'a1', 'a2', 'a3', 'a_idx', 't'])
    return df


def reform_dataset(directory, transcript_directory, split, way='riwa'):
    data = []
    correct_answer_list, incorrect_answer_list = [], []
    transcripts = get_transcripts_from_directory(transcript_directory)
    with open(os.path.join(directory, 'qa_' + split + '.json'), 'r') as f:
        json_data = [json.loads(line) for line in f]
    # collect all correct and incorrect answers
    for item in json_data:
        correct_answer_list.append(item['a'+str(item['answer_idx'])])
        for i in range(4):
            if i == item['answer_idx']:
                continue
            incorrect_answer_list.append(item['a'+str(i)])
    # reform the dataset using only correct answers
    for item in json_data:
        vid = item['qid'].split('_')[0]
        if vid in transcripts:
            transcript = ' '.join(transcripts[vid])
            if way == 'riwa':
                current_correct_answer = item['a'+str(item['answer_idx'])]
                incorrect_answers = random.sample(correct_answer_list, k=3)
                while current_correct_answer in incorrect_answers:
                    incorrect_answers = random.sample(correct_answer_list, k=3)
                current_data = copy.deepcopy(incorrect_answers)
                current_data.insert(int(item['answer_idx']), current_correct_answer)
                incorrect_answers.insert(int(item['answer_idx']), current_correct_answer)
                current_data.insert(0, item['q'])
                current_data.append(item['answer_idx'])
                current_data.append(transcript)
                current_data.append(item['q'] + incorrect_answers[0])
                current_data.append(item['q'] + incorrect_answers[1])
                current_data.append(item['q'] + incorrect_answers[2])
                current_data.append(item['q'] + incorrect_answers[3])
                current_data.append(item['q'] + incorrect_answers[0] + transcript)
                current_data.append(item['q'] + incorrect_answers[1] + transcript)
                current_data.append(item['q'] + incorrect_answers[2] + transcript)
                current_data.append(item['q'] + incorrect_answers[3] + transcript)
                data.append(current_data)
            elif way == 'riwi':
                current_correct_answer = item['a' + str(item['answer_idx'])]
                incorrect_answers = random.sample(incorrect_answer_list, k=3)
                current_data = copy.deepcopy(incorrect_answers)
                current_data.insert(int(item['answer_idx']), current_correct_answer)
                incorrect_answers.insert(int(item['answer_idx']), current_correct_answer)
                current_data.insert(0, item['q'])
                current_data.append(item['answer_idx'])
                current_data.append(transcript)
                current_data.append(item['q'] + incorrect_answers[0])
                current_data.append(item['q'] + incorrect_answers[1])
                current_data.append(item['q'] + incorrect_answers[2])
                current_data.append(item['q'] + incorrect_answers[3])
                current_data.append(item['q'] + incorrect_answers[0] + transcript)
                current_data.append(item['q'] + incorrect_answers[1] + transcript)
                current_data.append(item['q'] + incorrect_answers[2] + transcript)
                current_data.append(item['q'] + incorrect_answers[3] + transcript)
                data.append(current_data)
            elif way == 'rawi':
                current_correct_answer = random.sample(incorrect_answer_list, k=1)[0]
                incorrect_answers = []
                for i in range(4):
                    if i == item['answer_idx']:
                        continue
                    incorrect_answers.append(item['a' + str(i)])
                current_data = copy.deepcopy(incorrect_answers)
                current_data.insert(int(item['answer_idx']), current_correct_answer)
                incorrect_answers.insert(int(item['answer_idx']), current_correct_answer)
                current_data.insert(0, item['q'])
                current_data.append(item['answer_idx'])
                current_data.append(transcript)
                current_data.append(item['q'] + incorrect_answers[0])
                current_data.append(item['q'] + incorrect_answers[1])
                current_data.append(item['q'] + incorrect_answers[2])
                current_data.append(item['q'] + incorrect_answers[3])
                current_data.append(item['q'] + incorrect_answers[0] + transcript)
                current_data.append(item['q'] + incorrect_answers[1] + transcript)
                current_data.append(item['q'] + incorrect_answers[2] + transcript)
                current_data.append(item['q'] + incorrect_answers[3] + transcript)
                data.append(current_data)
            elif way == 'rawa':
                current_correct_answer = random.sample(correct_answer_list, k=1)[0]
                incorrect_answers = []
                for i in range(4):
                    if i == item['answer_idx']:
                        continue
                    incorrect_answers.append(item['a' + str(i)])
                current_data = copy.deepcopy(incorrect_answers)
                current_data.insert(int(item['answer_idx']), current_correct_answer)
                incorrect_answers.insert(int(item['answer_idx']), current_correct_answer)
                current_data.insert(0, item['q'])
                current_data.append(item['answer_idx'])
                current_data.append(transcript)
                current_data.append(item['q'] + incorrect_answers[0])
                current_data.append(item['q'] + incorrect_answers[1])
                current_data.append(item['q'] + incorrect_answers[2])
                current_data.append(item['q'] + incorrect_answers[3])
                current_data.append(item['q'] + incorrect_answers[0] + transcript)
                current_data.append(item['q'] + incorrect_answers[1] + transcript)
                current_data.append(item['q'] + incorrect_answers[2] + transcript)
                current_data.append(item['q'] + incorrect_answers[3] + transcript)
                data.append(current_data)

    df = pd.DataFrame(data, columns=['q', 'a0', 'a1', 'a2', 'a3', 'a_idx', 't',
                                     'qa0', 'qa1', 'qa2', 'qa3',
                                     'qta0', 'qta1', 'qta2', 'qta3'])
    return df


def calc_accuracy(predictions, labels):
    predictions_ = predictions.cpu().detach().numpy()
    labels_ = labels.cpu().detach().numpy()
    index = np.argmax(predictions_, axis=1)
    return (numpy.array(index) == numpy.array(labels_)).sum() / predictions.shape[0]


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    directory = "siq2/qa"
    transcript_directory = "siq2/transcript"
    if perturbation:
        trk = reform_dataset(directory, transcript_directory, 'train')
        dek = reform_dataset(directory, transcript_directory, 'val')
    else:
        trk = read_dataset(directory, transcript_directory, 'train')
        dek = read_dataset(directory, transcript_directory, 'val')

    tokenizer = T5Tokenizer.from_pretrained(t5_model_selection)
    # model = T5EncoderTextOnlyModel().to(device)
    model = T5EncoderTextOnlyModelDouble().to(device)
    judge_model = JudgeText().to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(judge_model.parameters()), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    dev_acc = []
    dev_acc_decrease_iters = 0
    for i in range(40):
        model.train()
        print("Epoch %d" % i)
        losses = []
        accs = []
        ds_size = len(trk)
        if ds_size % batch_size == 0:
            iters = int(ds_size / batch_size)
        else:
            iters = int(ds_size / batch_size) + 1
        for j in range(iters):
            this_trk = trk[j * batch_size:(j + 1) * batch_size]
            q = tokenizer(this_trk['q'].values.tolist(), return_tensors='pt', max_length=50,
                          truncation=True, padding='max_length').input_ids.to(device)
            t = tokenizer(this_trk['t'].values.tolist(), return_tensors='pt', max_length=500,
                          truncation=True, padding='max_length').input_ids.to(device)
            if separate:
                a0 = tokenizer(this_trk['a0'].values.tolist(), return_tensors='pt', max_length=50,
                               truncation=True, padding='max_length').input_ids.to(device)
                a1 = tokenizer(this_trk['a1'].values.tolist(), return_tensors='pt', max_length=50,
                               truncation=True, padding='max_length').input_ids.to(device)
                a2 = tokenizer(this_trk['a2'].values.tolist(), return_tensors='pt', max_length=50,
                               truncation=True, padding='max_length').input_ids.to(device)
                a3 = tokenizer(this_trk['a3'].values.tolist(), return_tensors='pt', max_length=50,
                               truncation=True, padding='max_length').input_ids.to(device)
                q, a0, a1, a2, a3, t = model(q, a0, a1, a2, a3, t)
            elif load_data == 'qac':
                qa0 = tokenizer(this_trk['qa0'].values.tolist(), return_tensors='pt', max_length=500,
                                truncation=True, padding='max_length').input_ids.to(device)
                qa1 = tokenizer(this_trk['qa1'].values.tolist(), return_tensors='pt', max_length=500,
                                truncation=True, padding='max_length').input_ids.to(device)
                qa2 = tokenizer(this_trk['qa2'].values.tolist(), return_tensors='pt', max_length=500,
                                truncation=True, padding='max_length').input_ids.to(device)
                qa3 = tokenizer(this_trk['qa3'].values.tolist(), return_tensors='pt', max_length=500,
                                truncation=True, padding='max_length').input_ids.to(device)
                q, qa0, qa1, qa2, qa3, t = model(q, qa0, qa1, qa2, qa3, t)
            else:
                qta0 = tokenizer(this_trk['qta0'].values.tolist(), return_tensors='pt', max_length=500,
                                 truncation=True, padding='max_length').input_ids.to(device)
                qta1 = tokenizer(this_trk['qta1'].values.tolist(), return_tensors='pt', max_length=500,
                                 truncation=True, padding='max_length').input_ids.to(device)
                qta2 = tokenizer(this_trk['qta2'].values.tolist(), return_tensors='pt', max_length=500,
                                 truncation=True, padding='max_length').input_ids.to(device)
                qta3 = tokenizer(this_trk['qta3'].values.tolist(), return_tensors='pt', max_length=500,
                                 truncation=True, padding='max_length').input_ids.to(device)
                q, qta0, qta1, qta2, qta3, t = model(q, qta0, qta1, qta2, qta3, t)

            labels = torch.LongTensor(this_trk['a_idx'].values.tolist()).to(device)

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

            optimizer.zero_grad()
            loss = loss_fn(a, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            accs.append(calc_accuracy(a, labels))

        print("Loss %f", numpy.array(losses, dtype="float32").mean())
        print("Accs %f", numpy.array(accs, dtype="float32").mean())

        _accs = []
        model.eval()
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
        acc_temp = numpy.array(_accs, dtype="float32").mean()
        if acc_temp > best_acc:
            best_acc = acc_temp
            torch.save({
                'epoch': i,
                'modelA_state_dict': model.state_dict(),
                'modelB_state_dict': judge_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'best.pt')
        if len(dev_acc) > 0 and dev_acc[-1] > acc_temp:
            dev_acc_decrease_iters += 1
        else:
            dev_acc_decrease_iters = 0
        dev_acc.append(acc_temp)
        if dev_acc_decrease_iters > 4:
            break
    print(best_acc)


