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

from transformers import set_seed
from transformers import T5Tokenizer, BertTokenizer
from transformers import LongT5EncoderModel
from transformers import ViTModel, ViTImageProcessor
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import WhisperProcessor

import random
import json
import pandas as pd
import math

from utils.get_statistics import *
from text_only_model import T5EncoderTextOnlyModel, TextEncoder
from text_only_model import calc_accuracy
from PIL import Image
from scipy.io import wavfile
import scipy.signal as sps


set_seed(42)
logger = logging.getLogger(__name__)

# parameters
t5_model_selection = 't5-small'
t5_output_dim = 512
vit_output_dim = 768

long_t5_model_selection = 'google/long-t5-local-base'
long_t5_output_dim = 512
bert_model_selection = 'bert-base-uncased'
bert_output_dim = 768

load_data = 'qsac'
if 'c' in load_data:
    separate = False
else:
    separate = True
if load_data == 'ia' or load_data == 'sa':
    output_dim = t5_output_dim + vit_output_dim
elif load_data == 'qia' or load_data == 'qsa':
    output_dim = 2 * t5_output_dim + vit_output_dim
elif load_data == 'qtia':
    output_dim = 3 * t5_output_dim + vit_output_dim
elif load_data == 'qtisa':
    output_dim = 3 * t5_output_dim + 2 * vit_output_dim
elif load_data == 'qac':
    output_dim = t5_output_dim * 2
elif load_data == 'qiac' or load_data == 'qsac' or load_data == 'qtac':
    output_dim = t5_output_dim * 3
elif load_data == 'qisac':
    output_dim = t5_output_dim * 4
elif load_data == 'qtisac':
    output_dim = t5_output_dim * 5

perturbation = False

transcript_num = 50
# vision model
vit_model_selection = 'google/vit-base-patch16-224-in21k'
# integer, larger numbers represent small number of images
image_sample_rate = 3
image_num = math.floor(182 / image_sample_rate)

# audio model
wav2vec_model_selection = 'facebook/wav2vec2-base-960h'
whisper_model_selection = 'openai/whisper-medium.en'

batch_size = 1


class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        self.vision_model = ViTModel.from_pretrained(vit_model_selection)
        self.avg_pooling = nn.AvgPool2d(kernel_size=(image_num, 197))

    def forward(self, x):
        x = self.vision_model(**x).last_hidden_state
        x = x.view(-1, image_num, x.shape[1], x.shape[2])
        x = x.permute(0, 3, 1, 2)
        x = self.avg_pooling(x)
        return x


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.vision_model = ViTModel.from_pretrained(vit_model_selection)
        self.avg_pooling = nn.AvgPool2d(kernel_size=(1, 197))

    def forward(self, x):
        x = self.vision_model(**x).last_hidden_state
        x = x.view(-1, image_num, x.shape[1], x.shape[2])
        x = x.permute(0, 3, 1, 2)
        x = self.avg_pooling(x)
        return x.squeeze(3).permute(0, 2, 1)


class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()
        self.audio_model = Wav2Vec2Model.from_pretrained(wav2vec_model_selection)
        self.avg_pooling = nn.AvgPool1d(kernel_size=50)

    def forward(self, x):
        x = self.audio_model(**x).last_hidden_state
        x = self.avg_pooling(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class CombineModel(nn.Module):
    def __init__(self):
        super(CombineModel, self).__init__()
        self.base_model = LongT5EncoderModel.from_pretrained(t5_model_selection)

    def forward(self, x):
        x = self.base_model(inputs_embeds=x).last_hidden_state
        return x


class JudgeTextAndVision(nn.Module):
    def __init__(self):
        super(JudgeTextAndVision, self).__init__()
        self.linear1 = nn.Linear(output_dim, 25)
        self.activation = torch.nn.Sigmoid()
        self.linear2 = nn.Linear(25, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        return x


def get_images_from_directory(image_directory, vid_list=None):
    img_dictionary = {}
    for index, dirname in enumerate(sorted(os.listdir(image_directory))):
        if vid_list and dirname not in vid_list:
            continue
        subdir = os.path.join(image_directory, dirname)
        if os.path.isdir(subdir):
            img_list = []
            for filename in os.listdir(subdir):
                img = Image.open(os.path.join(subdir, filename))
                img_list.append(np.asarray(img))
            img_dictionary.update({dirname: img_list[::image_sample_rate][:image_num]})
    return img_dictionary


def get_audios_from_directory(audio_directory, new_rate=16000, vid_list=None):
    audio_dictionary = {}
    for index, filename in enumerate(sorted(os.listdir(audio_directory))):
        if vid_list and filename[:-4] not in vid_list:
            continue
        audio_file = os.path.join(audio_directory, filename)
        if audio_file.endswith('.wav') or audio_file.endswith('.mp3'):
            sampling_rate, audio_array = wavfile.read(audio_file)
            # Resample data
            number_of_samples = round(len(audio_array) * float(new_rate) / sampling_rate)
            audio_array = sps.resample(audio_array, number_of_samples)
            # Convert int to float32
            audio_array = audio_array.astype(np.float32, order='C') / 32768.0
            audio_dictionary.update({filename[:-4]: audio_array})
    return audio_dictionary


def read_dataset(directory, transcripts, images, audios, split):
    data = []
    with open(os.path.join(directory, 'qa_' + split + '.json'), 'r') as f:
        json_data = [json.loads(line) for line in f]
    for item in json_data:
        vid = item['qid'].split('_')[0]
        if vid in transcripts and vid in images:
            transcript = ' '.join(transcripts[vid])
            image = images[vid]
            audio = audios[vid]
            data.append([item['q'], item['a0'], item['a1'], item['a2'], item['a3'], item['answer_idx'],
                         transcript, image, audio])
    df = pd.DataFrame(data, columns=['q', 'a0', 'a1', 'a2', 'a3', 'a_idx', 't', 'i', 's'])
    return df


def reform_dataset(directory, transcripts, images, audios, split):
    data = []
    correct_answer_list = []
    with open(os.path.join(directory, 'qa_' + split + '.json'), 'r') as f:
        json_data = [json.loads(line) for line in f]
    # collect all correct answers
    for item in json_data:
        correct_answer_list.append(item['a'+str(item['answer_idx'])])
    # reform the dataset using only correct answers
    for item in json_data:
        vid = item['qid'].split('_')[0]
        if vid in transcripts and vid in images:
            transcript = ' '.join(transcripts[vid])
            image = images[vid]
            audio = audios[vid]
            current_correct_answer = item['a'+str(item['answer_idx'])]
            incorrect_answers = random.sample(correct_answer_list, k=3)
            while current_correct_answer in incorrect_answers:
                incorrect_answers = random.sample(correct_answer_list, k=3)
            incorrect_answers.insert(int(item['answer_idx']), current_correct_answer)
            incorrect_answers.insert(0, item['q'])
            incorrect_answers.append(item['answer_idx'])
            incorrect_answers.append(transcript)
            incorrect_answers.append(image)
            incorrect_answers.append(audio)
            data.append(incorrect_answers)

    df = pd.DataFrame(data, columns=['q', 'a0', 'a1', 'a2', 'a3', 'a_idx', 't', 'i', 's'])
    return df


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    directory = "siq2/qa"
    transcript_directory = "siq2/transcript"
    image_directory = "siq2/frames"
    audio_directory = "siq2/audio/wav"

    transcript_dictionary = get_transcripts_from_directory(transcript_directory)
    image_dictionary = get_images_from_directory(image_directory)
    audio_dictionary = get_audios_from_directory(audio_directory)

    if perturbation:
        trk = reform_dataset(directory, transcript_dictionary, image_dictionary, audio_dictionary, 'train')
        dek = reform_dataset(directory, transcript_dictionary, image_dictionary, audio_dictionary, 'val')
    else:
        trk = read_dataset(directory, transcript_dictionary, image_dictionary, audio_dictionary, 'train')
        dek = read_dataset(directory, transcript_dictionary, image_dictionary, audio_dictionary, 'val')

    image_processor = ViTImageProcessor.from_pretrained(vit_model_selection)
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_selection)
    judge_model = JudgeTextAndVision().to(device)

    if separate:
        tokenizer = T5Tokenizer.from_pretrained(t5_model_selection)
        text_model = T5EncoderTextOnlyModel().to(device)
        image_model = VisionModel().to(device)
        audio_model = Wav2Vec2Model.from_pretrained(wav2vec_model_selection).to(device)
        optimizer = optim.AdamW(list(text_model.parameters()) + list(image_model.parameters()) +
                                list(judge_model.parameters()) + list(audio_model.parameters()), lr=1e-4)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_model_selection)
        text_model = TextEncoder().to(device)
        image_model = ImageModel().to(device)
        audio_model = AudioModel().to(device)

        transcript_pooling = nn.AvgPool1d(kernel_size=10).to(device)
        text_projector = nn.Linear(bert_output_dim, t5_output_dim).to(device)
        image_projector = nn.Linear(bert_output_dim, t5_output_dim).to(device)
        audio_projector = nn.Linear(bert_output_dim, t5_output_dim).to(device)

        combine_model = CombineModel().to(device)

        optimizer = optim.AdamW(list(combine_model.parameters()) + list(judge_model.parameters()) +
                                list(text_projector.parameters()) + list(image_projector.parameters()) +
                                list(audio_projector.parameters()),
                                lr=1e-4)

    # whisper_processor = WhisperProcessor.from_pretrained(whisper_model_selection)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    dev_acc = []
    dev_acc_decrease_iters = 0
    for i in range(40):
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
            if separate:
                q = tokenizer(this_trk['q'].values.tolist(), return_tensors='pt', max_length=50,
                              truncation=True, padding='max_length').input_ids.to(device)
                a0 = tokenizer(this_trk['a0'].values.tolist(), return_tensors='pt', max_length=50,
                               truncation=True, padding='max_length').input_ids.to(device)
                a1 = tokenizer(this_trk['a1'].values.tolist(), return_tensors='pt', max_length=50,
                               truncation=True, padding='max_length').input_ids.to(device)
                a2 = tokenizer(this_trk['a2'].values.tolist(), return_tensors='pt', max_length=50,
                               truncation=True, padding='max_length').input_ids.to(device)
                a3 = tokenizer(this_trk['a3'].values.tolist(), return_tensors='pt', max_length=50,
                               truncation=True, padding='max_length').input_ids.to(device)
                t = tokenizer(this_trk['t'].values.tolist(), return_tensors='pt', max_length=500,
                              truncation=True, padding='max_length').input_ids.to(device)
                i = [item for sublist in this_trk['i'].values for item in sublist]
                i = image_processor(i, return_tensors='pt').to(device)
                s = wav2vec_processor(this_trk['s'].values.tolist(), sampling_rate=16000,
                                      return_tensors='pt').to(device)

                q, a0, a1, a2, a3, t = text_model(q, a0, a1, a2, a3, t)
                i = image_model(i).squeeze(3).squeeze(2)
                s = audio_model(**s).last_hidden_state[:, -1, :]
            else:
                if 'q' in load_data:
                    q = tokenizer(this_trk['q'].values.tolist(), return_tensors='pt', max_length=50,
                                  truncation=True, padding='max_length').input_ids.to(device)
                    q = text_projector(text_model(q))
                if 'a' in load_data:
                    a0 = tokenizer(this_trk['a0'].values.tolist(), return_tensors='pt', max_length=50,
                                   truncation=True, padding='max_length').input_ids.to(device)
                    a1 = tokenizer(this_trk['a1'].values.tolist(), return_tensors='pt', max_length=50,
                                   truncation=True, padding='max_length').input_ids.to(device)
                    a2 = tokenizer(this_trk['a2'].values.tolist(), return_tensors='pt', max_length=50,
                                   truncation=True, padding='max_length').input_ids.to(device)
                    a3 = tokenizer(this_trk['a3'].values.tolist(), return_tensors='pt', max_length=50,
                                   truncation=True, padding='max_length').input_ids.to(device)
                    a0 = text_projector(text_model(a0))
                    a1 = text_projector(text_model(a1))
                    a2 = text_projector(text_model(a2))
                    a3 = text_projector(text_model(a3))
                if 't' in load_data:
                    t = tokenizer(this_trk['t'].values.tolist(), return_tensors='pt', max_length=500,
                                  truncation=True, padding='max_length').input_ids.to(device)
                    t = text_projector(transcript_pooling(text_model(t).permute(0, 2, 1)).permute(0, 2, 1))
                if 'i' in load_data:
                    i = [item for sublist in this_trk['i'].values for item in sublist]
                    i = image_processor(i, return_tensors='pt').to(device)
                    i = image_projector(image_model(i))
                if 's' in load_data:
                    s = wav2vec_processor(this_trk['s'].values.tolist(), sampling_rate=16000,
                                          return_tensors='pt').to(device)
                    s = audio_projector(audio_model(s))

            labels = torch.LongTensor(this_trk['a_idx'].values.tolist()).to(device)

            if load_data == 'ia':
                a0 = judge_model(torch.cat((i, a0), dim=1))
                a1 = judge_model(torch.cat((i, a1), dim=1))
                a2 = judge_model(torch.cat((i, a2), dim=1))
                a3 = judge_model(torch.cat((i, a3), dim=1))
            elif load_data == 'sa':
                a0 = judge_model(torch.cat((s, a0), dim=1))
                a1 = judge_model(torch.cat((s, a1), dim=1))
                a2 = judge_model(torch.cat((s, a2), dim=1))
                a3 = judge_model(torch.cat((s, a3), dim=1))
            elif load_data == 'qia':
                a0 = judge_model(torch.cat((q, i, a0), dim=1))
                a1 = judge_model(torch.cat((q, i, a1), dim=1))
                a2 = judge_model(torch.cat((q, i, a2), dim=1))
                a3 = judge_model(torch.cat((q, i, a3), dim=1))
            elif load_data == 'qsa':
                a0 = judge_model(torch.cat((q, s, a0), dim=1))
                a1 = judge_model(torch.cat((q, s, a1), dim=1))
                a2 = judge_model(torch.cat((q, s, a2), dim=1))
                a3 = judge_model(torch.cat((q, s, a3), dim=1))
            elif load_data == 'qtia':
                a0 = judge_model(torch.cat((q, t, i, a0), dim=1))
                a1 = judge_model(torch.cat((q, t, i, a1), dim=1))
                a2 = judge_model(torch.cat((q, t, i, a2), dim=1))
                a3 = judge_model(torch.cat((q, t, i, a3), dim=1))
            elif load_data == 'qtsa':
                a0 = judge_model(torch.cat((q, t, s, a0), dim=1))
                a1 = judge_model(torch.cat((q, t, s, a1), dim=1))
                a2 = judge_model(torch.cat((q, t, s, a2), dim=1))
                a3 = judge_model(torch.cat((q, t, s, a3), dim=1))
            elif load_data == 'qisa':
                a0 = judge_model(torch.cat((q, i, s, a0), dim=1))
                a1 = judge_model(torch.cat((q, i, s, a1), dim=1))
                a2 = judge_model(torch.cat((q, i, s, a2), dim=1))
                a3 = judge_model(torch.cat((q, i, s, a3), dim=1))
            elif load_data == 'qtisa':
                a0 = judge_model(torch.cat((q, t, i, s, a0), dim=1))
                a1 = judge_model(torch.cat((q, t, i, s, a1), dim=1))
                a2 = judge_model(torch.cat((q, t, i, s, a2), dim=1))
                a3 = judge_model(torch.cat((q, t, i, s, a3), dim=1))
            elif load_data == 'qac':
                a0 = judge_model(combine_model(torch.cat((q, a0), dim=1))[:, [49, -1], :].view(-1, output_dim))
                a1 = judge_model(combine_model(torch.cat((q, a1), dim=1))[:, [49, -1], :].view(-1, output_dim))
                a2 = judge_model(combine_model(torch.cat((q, a2), dim=1))[:, [49, -1], :].view(-1, output_dim))
                a3 = judge_model(combine_model(torch.cat((q, a3), dim=1))[:, [49, -1], :].view(-1, output_dim))
            elif load_data == 'qtac':
                a0 = judge_model(
                    combine_model(torch.cat((q, t, a0), dim=1))[:, [49, 49 + transcript_num, -1], :]
                    .view(-1, output_dim))
                a1 = judge_model(
                    combine_model(torch.cat((q, t, a1), dim=1))[:, [49, 49 + transcript_num, -1], :]
                    .view(-1, output_dim))
                a2 = judge_model(
                    combine_model(torch.cat((q, t, a2), dim=1))[:, [49, 49 + transcript_num, -1], :]
                    .view(-1, output_dim))
                a3 = judge_model(
                    combine_model(torch.cat((q, t, a3), dim=1))[:, [49, 49 + transcript_num, -1], :]
                    .view(-1, output_dim))
            elif load_data == 'qiac':
                a0 = judge_model(
                    combine_model(torch.cat((q, i, a0), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
                a1 = judge_model(
                    combine_model(torch.cat((q, i, a1), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
                a2 = judge_model(
                    combine_model(torch.cat((q, i, a2), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
                a3 = judge_model(
                    combine_model(torch.cat((q, i, a3), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
            elif load_data == 'qsac':
                a0 = judge_model(
                    combine_model(torch.cat((q, s, a0), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
                a1 = judge_model(
                    combine_model(torch.cat((q, s, a1), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
                a2 = judge_model(
                    combine_model(torch.cat((q, s, a2), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
                a3 = judge_model(
                    combine_model(torch.cat((q, s, a3), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
            elif load_data == 'qisac':
                a0 = judge_model(combine_model(torch.cat((q, i, s, a0), dim=1))[:,
                                 [49, 49 + image_num, 49 + 2 * image_num, -1], :].view(-1, output_dim))
                a1 = judge_model(combine_model(torch.cat((q, i, s, a1), dim=1))[:,
                                 [49, 49 + image_num, 49 + 2 * image_num, -1], :].view(-1, output_dim))
                a2 = judge_model(combine_model(torch.cat((q, i, s, a2), dim=1))[:,
                                 [49, 49 + image_num, 49 + 2 * image_num, -1], :].view(-1, output_dim))
                a3 = judge_model(combine_model(torch.cat((q, i, s, a3), dim=1))[:,
                                 [49, 49 + image_num, 49 + 2 * image_num, -1], :].view(-1, output_dim))
            elif load_data == 'qtisac':
                a0 = judge_model(combine_model(torch.cat((q, t, i, s, a0), dim=1))[:,
                                 [49, 49 + transcript_num, 49 + transcript_num + image_num,
                                  49 + transcript_num + 2 * image_num, -1], :].view(-1, output_dim))
                a1 = judge_model(combine_model(torch.cat((q, t, i, s, a1), dim=1))[:,
                                 [49, 49 + transcript_num, 49 + transcript_num + image_num,
                                  49 + transcript_num + 2 * image_num, -1], :].view(-1, output_dim))
                a2 = judge_model(combine_model(torch.cat((q, t, i, s, a2), dim=1))[:,
                                 [49, 49 + transcript_num, 49 + transcript_num + image_num,
                                  49 + transcript_num + 2 * image_num, -1], :].view(-1, output_dim))
                a3 = judge_model(combine_model(torch.cat((q, t, i, s, a3), dim=1))[:,
                                 [49, 49 + transcript_num, 49 + transcript_num + image_num,
                                  49 + transcript_num + 2 * image_num, -1], :].view(-1, output_dim))
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
        ds_size = len(dek)
        if ds_size % batch_size == 0:
            iters = int(ds_size / batch_size)
        else:
            iters = int(ds_size / batch_size) + 1
        for j in range(iters):
            this_dek = dek[j * batch_size:(j + 1) * batch_size]
            if separate:
                q = tokenizer(this_dek['q'].values.tolist(), return_tensors='pt', max_length=50,
                              truncation=True, padding='max_length').input_ids.to(device)
                a0 = tokenizer(this_dek['a0'].values.tolist(), return_tensors='pt', max_length=50,
                               truncation=True, padding='max_length').input_ids.to(device)
                a1 = tokenizer(this_dek['a1'].values.tolist(), return_tensors='pt', max_length=50,
                               truncation=True, padding='max_length').input_ids.to(device)
                a2 = tokenizer(this_dek['a2'].values.tolist(), return_tensors='pt', max_length=50,
                               truncation=True, padding='max_length').input_ids.to(device)
                a3 = tokenizer(this_dek['a3'].values.tolist(), return_tensors='pt', max_length=50,
                               truncation=True, padding='max_length').input_ids.to(device)
                t = tokenizer(this_dek['t'].values.tolist(), return_tensors='pt', max_length=500,
                              truncation=True, padding='max_length').input_ids.to(device)
                i = [item for sublist in this_dek['i'].values for item in sublist]
                i = image_processor(i, return_tensors='pt').to(device)
                s = wav2vec_processor(this_dek['s'].values.tolist(), sampling_rate=16000,
                                      return_tensors='pt').to(device)

                q, a0, a1, a2, a3, t = text_model(q, a0, a1, a2, a3, t)
                i = image_model(i).squeeze(3).squeeze(2)
                s = audio_model(**s).last_hidden_state[:, -1, :]
            else:
                if 'q' in load_data:
                    q = tokenizer(this_dek['q'].values.tolist(), return_tensors='pt', max_length=50,
                                  truncation=True, padding='max_length').input_ids.to(device)
                    q = text_projector(text_model(q))
                if 'a' in load_data:
                    a0 = tokenizer(this_dek['a0'].values.tolist(), return_tensors='pt', max_length=50,
                                   truncation=True, padding='max_length').input_ids.to(device)
                    a1 = tokenizer(this_dek['a1'].values.tolist(), return_tensors='pt', max_length=50,
                                   truncation=True, padding='max_length').input_ids.to(device)
                    a2 = tokenizer(this_dek['a2'].values.tolist(), return_tensors='pt', max_length=50,
                                   truncation=True, padding='max_length').input_ids.to(device)
                    a3 = tokenizer(this_dek['a3'].values.tolist(), return_tensors='pt', max_length=50,
                                   truncation=True, padding='max_length').input_ids.to(device)
                    a0 = text_projector(text_model(a0))
                    a1 = text_projector(text_model(a1))
                    a2 = text_projector(text_model(a2))
                    a3 = text_projector(text_model(a3))
                if 't' in load_data:
                    t = tokenizer(this_dek['t'].values.tolist(), return_tensors='pt', max_length=500,
                                  truncation=True, padding='max_length').input_ids.to(device)
                    t = text_projector(transcript_pooling(text_model(t).permute(0, 2, 1)).permute(0, 2, 1))
                if 'i' in load_data:
                    i = [item for sublist in this_dek['i'].values for item in sublist]
                    i = image_processor(i, return_tensors='pt').to(device)
                    i = image_projector(image_model(i))
                if 's' in load_data:
                    s = wav2vec_processor(this_dek['s'].values.tolist(), sampling_rate=16000,
                                          return_tensors='pt').to(device)
                    s = audio_projector(audio_model(s))

            labels = torch.LongTensor(this_dek['a_idx'].values.tolist()).to(device)

            if load_data == 'ia':
                a0 = judge_model(torch.cat((i, a0), dim=1))
                a1 = judge_model(torch.cat((i, a1), dim=1))
                a2 = judge_model(torch.cat((i, a2), dim=1))
                a3 = judge_model(torch.cat((i, a3), dim=1))
            elif load_data == 'sa':
                a0 = judge_model(torch.cat((s, a0), dim=1))
                a1 = judge_model(torch.cat((s, a1), dim=1))
                a2 = judge_model(torch.cat((s, a2), dim=1))
                a3 = judge_model(torch.cat((s, a3), dim=1))
            elif load_data == 'qia':
                a0 = judge_model(torch.cat((q, i, a0), dim=1))
                a1 = judge_model(torch.cat((q, i, a1), dim=1))
                a2 = judge_model(torch.cat((q, i, a2), dim=1))
                a3 = judge_model(torch.cat((q, i, a3), dim=1))
            elif load_data == 'qsa':
                a0 = judge_model(torch.cat((q, s, a0), dim=1))
                a1 = judge_model(torch.cat((q, s, a1), dim=1))
                a2 = judge_model(torch.cat((q, s, a2), dim=1))
                a3 = judge_model(torch.cat((q, s, a3), dim=1))
            elif load_data == 'qtia':
                a0 = judge_model(torch.cat((q, t, i, a0), dim=1))
                a1 = judge_model(torch.cat((q, t, i, a1), dim=1))
                a2 = judge_model(torch.cat((q, t, i, a2), dim=1))
                a3 = judge_model(torch.cat((q, t, i, a3), dim=1))
            elif load_data == 'qtsa':
                a0 = judge_model(torch.cat((q, t, s, a0), dim=1))
                a1 = judge_model(torch.cat((q, t, s, a1), dim=1))
                a2 = judge_model(torch.cat((q, t, s, a2), dim=1))
                a3 = judge_model(torch.cat((q, t, s, a3), dim=1))
            elif load_data == 'qisa':
                a0 = judge_model(torch.cat((q, i, s, a0), dim=1))
                a1 = judge_model(torch.cat((q, i, s, a1), dim=1))
                a2 = judge_model(torch.cat((q, i, s, a2), dim=1))
                a3 = judge_model(torch.cat((q, i, s, a3), dim=1))
            elif load_data == 'qtisa':
                a0 = judge_model(torch.cat((q, t, i, s, a0), dim=1))
                a1 = judge_model(torch.cat((q, t, i, s, a1), dim=1))
                a2 = judge_model(torch.cat((q, t, i, s, a2), dim=1))
                a3 = judge_model(torch.cat((q, t, i, s, a3), dim=1))
            elif load_data == 'qac':
                a0 = judge_model(combine_model(torch.cat((q, a0), dim=1))[:, [49, -1], :].view(-1, output_dim))
                a1 = judge_model(combine_model(torch.cat((q, a1), dim=1))[:, [49, -1], :].view(-1, output_dim))
                a2 = judge_model(combine_model(torch.cat((q, a2), dim=1))[:, [49, -1], :].view(-1, output_dim))
                a3 = judge_model(combine_model(torch.cat((q, a3), dim=1))[:, [49, -1], :].view(-1, output_dim))
            elif load_data == 'qtac':
                a0 = judge_model(
                    combine_model(torch.cat((q, t, a0), dim=1))[:, [49, 49 + transcript_num, -1], :]
                    .view(-1, output_dim))
                a1 = judge_model(
                    combine_model(torch.cat((q, t, a1), dim=1))[:, [49, 49 + transcript_num, -1], :]
                    .view(-1, output_dim))
                a2 = judge_model(
                    combine_model(torch.cat((q, t, a2), dim=1))[:, [49, 49 + transcript_num, -1], :]
                    .view(-1, output_dim))
                a3 = judge_model(
                    combine_model(torch.cat((q, t, a3), dim=1))[:, [49, 49 + transcript_num, -1], :]
                    .view(-1, output_dim))
            elif load_data == 'qiac':
                a0 = judge_model(
                    combine_model(torch.cat((q, i, a0), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
                a1 = judge_model(
                    combine_model(torch.cat((q, i, a1), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
                a2 = judge_model(
                    combine_model(torch.cat((q, i, a2), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
                a3 = judge_model(
                    combine_model(torch.cat((q, i, a3), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
            elif load_data == 'qsac':
                a0 = judge_model(
                    combine_model(torch.cat((q, s, a0), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
                a1 = judge_model(
                    combine_model(torch.cat((q, s, a1), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
                a2 = judge_model(
                    combine_model(torch.cat((q, s, a2), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
                a3 = judge_model(
                    combine_model(torch.cat((q, s, a3), dim=1))[:, [49, 49 + image_num, -1], :].view(-1, output_dim))
            elif load_data == 'qisac':
                a0 = judge_model(combine_model(torch.cat((q, i, s, a0), dim=1))[:,
                                 [49, 49 + image_num, 49 + 2 * image_num, -1], :].view(-1, output_dim))
                a1 = judge_model(combine_model(torch.cat((q, i, s, a1), dim=1))[:,
                                 [49, 49 + image_num, 49 + 2 * image_num, -1], :].view(-1, output_dim))
                a2 = judge_model(combine_model(torch.cat((q, i, s, a2), dim=1))[:,
                                 [49, 49 + image_num, 49 + 2 * image_num, -1], :].view(-1, output_dim))
                a3 = judge_model(combine_model(torch.cat((q, i, s, a3), dim=1))[:,
                                 [49, 49 + image_num, 49 + 2 * image_num, -1], :].view(-1, output_dim))
            elif load_data == 'qtisac':
                a0 = judge_model(combine_model(torch.cat((q, t, i, s, a0), dim=1))[:,
                                 [49, 49 + transcript_num, 49 + transcript_num + image_num,
                                  49 + transcript_num + 2 * image_num, -1], :].view(-1, output_dim))
                a1 = judge_model(combine_model(torch.cat((q, t, i, s, a1), dim=1))[:,
                                 [49, 49 + transcript_num, 49 + transcript_num + image_num,
                                  49 + transcript_num + 2 * image_num, -1], :].view(-1, output_dim))
                a2 = judge_model(combine_model(torch.cat((q, t, i, s, a2), dim=1))[:,
                                 [49, 49 + transcript_num, 49 + transcript_num + image_num,
                                  49 + transcript_num + 2 * image_num, -1], :].view(-1, output_dim))
                a3 = judge_model(combine_model(torch.cat((q, t, i, s, a3), dim=1))[:,
                                 [49, 49 + transcript_num, 49 + transcript_num + image_num,
                                  49 + transcript_num + 2 * image_num, -1], :].view(-1, output_dim))
            a = torch.cat((a0, a1, a2, a3), dim=1)
            _accs.append(calc_accuracy(a, labels))

        print("Dev Accs %f", numpy.array(_accs, dtype="float32").mean())
        print("-----------")
        acc_temp = numpy.array(_accs, dtype="float32").mean()
        if acc_temp > best_acc:
            best_acc = acc_temp
            if separate:
                torch.save({
                    'epoch': i,
                    'modelA_state_dict': text_model.state_dict(),
                    'modelB_state_dict': judge_model.state_dict(),
                    'modelC_state_dict': image_model.state_dict(),
                    'modelD_state_dict': audio_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'best.pt')
            else:
                torch.save({
                    'epoch': i,
                    'modelA_state_dict': combine_model.state_dict(),
                    'modelB_state_dict': judge_model.state_dict(),
                    'modelC_state_dict': text_projector.state_dict(),
                    'modelD_state_dict': image_projector.state_dict(),
                    'modelE_state_dict': audio_projector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'combine_best.pt')
        if len(dev_acc) > 0 and dev_acc[-1] > acc_temp:
            dev_acc_decrease_iters += 1
        else:
            dev_acc_decrease_iters = 0
        dev_acc.append(acc_temp)
        if dev_acc_decrease_iters > 4:
            break
    print(best_acc)


