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
from transformers import ViTImageProcessor
from transformers import Wav2Vec2Processor, Wav2Vec2Model

import random
import json
import pandas as pd
import math

from utils.get_statistics import get_transcripts_from_directory
from text_only_model import T5EncoderTextOnlyModel, TextEncoder
from text_only_model import calc_accuracy
from multimodal_model import VisionModel, ImageModel, AudioModel, CombineModel, JudgeTextAndVision
from multimodal_model import read_dataset, reform_dataset, get_images_from_directory, get_audios_from_directory


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
is_test = False


def read_test_dataset(directory, transcripts, images, audios, split):
    data = []
    with open(os.path.join(directory, 'qa_' + split + '.json'), 'r') as f:
        json_data = [json.loads(line) for line in f]
    for item in json_data:
        vid = item['qid'].split('_')[0]
        if vid in transcripts and vid in images:
            transcript = ' '.join(transcripts[vid])
            image = images[vid]
            audio = audios[vid]
            data.append([item['qid'], item['q'], item['a0'], item['a1'], item['a2'], item['a3'],
                         transcript, image, audio])
    df = pd.DataFrame(data, columns=['qid', 'q', 'a0', 'a1', 'a2', 'a3', 't', 'i', 's'])
    return df


def read_full_dataset(directory, split):
    with open(os.path.join(directory, 'qa_' + split + '.json'), 'r') as f:
        json_data = [json.loads(line) for line in f]
    return json_data


def save_full_dataset(data, prediction, directory, split):
    for item in data:
        qid = item['qid']
        if qid in prediction:
            item['answer_idx'] = prediction[qid]
    with open(os.path.join(directory, 'qa_' + split + '.json'), 'w') as f:
        json.dump(data, f)


def get_prediction(predictions):
    predictions_ = predictions.cpu().detach().numpy()
    index = np.argmax(predictions_, axis=1)
    return index.tolist()


def return_vid(directory, split):
    vid_list = []
    with open(os.path.join(directory, 'qa_' + split + '.json'), 'r') as f:
        json_data = [json.loads(line) for line in f]
        for item in json_data:
            vid = item['vid_name']
            if vid not in vid_list:
                vid_list.append(vid)
    return vid_list


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    directory = "siq2/qa"
    transcript_directory = "siq2/transcript"
    image_directory = "siq2/frames"
    audio_directory = "siq2/audio/wav"

    vid_list = return_vid(directory, 'test_human_annotated')

    transcript_dictionary = get_transcripts_from_directory(transcript_directory, vid_list)
    image_dictionary = get_images_from_directory(image_directory, vid_list)
    audio_dictionary = get_audios_from_directory(audio_directory, vid_list=vid_list)

    if perturbation:
        dek = reform_dataset(directory, transcript_dictionary, image_dictionary, audio_dictionary, 'val')
    else:
        dek = read_dataset(directory, transcript_dictionary, image_dictionary, audio_dictionary, 'test_human_annotated')

    if is_test:
        dek = read_test_dataset(directory, transcript_dictionary, image_dictionary, audio_dictionary, 'test')

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

        checkpoint = torch.load('best.pt')
        text_model.load_state_dict(checkpoint['modelA_state_dict'])
        judge_model.load_state_dict(checkpoint['modelB_state_dict'])
        image_model.load_state_dict(checkpoint['modelC_state_dict'])
        audio_model.load_state_dict(checkpoint['modelD_state_dict'])
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

        checkpoint = torch.load('combine_best.pt')
        combine_model.load_state_dict(checkpoint['modelA_state_dict'])
        judge_model.load_state_dict(checkpoint['modelB_state_dict'])
        text_projector.load_state_dict(checkpoint['modelC_state_dict'])
        image_projector.load_state_dict(checkpoint['modelD_state_dict'])
        audio_projector.load_state_dict(checkpoint['modelE_state_dict'])


    _accs = []
    qid_prediction_dic = {}
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

        if not is_test:
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
        if is_test:
            question_id = this_dek['qid'].values.tolist()
            predictions = get_prediction(a)
            for (id, prediction) in zip(question_id, predictions):
                qid_prediction_dic.update({id: prediction})
        else:
            _accs.append(calc_accuracy(a, labels))

    if is_test:
        data = read_full_dataset(directory, 'test')
        save_full_dataset(data, qid_prediction_dic, directory, 'test_predict')
    else:
        print("Dev Accs %f", numpy.array(_accs, dtype="float32").mean())
        print("-----------")


