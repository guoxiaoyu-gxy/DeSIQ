import pandas as pd
import logging
from utils import dataset_utils, constants
import os
os.environ['TRANSFORMERS_CACHE'] = constants.TRANSFORMERS_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = constants.DATASETS_CACHE_DIR

import numpy
from mmsdk import mmdatasdk
from mmsdk.mmmodelsdk.fusion.tensor_fusion.model import TensorFusion as TensorFusion
import socialiq.mylstm as mylstm
from visulizations.tsne_visualize import *

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from transformers import set_seed
from transformers import T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer
from transformers import Seq2SeqTrainingArguments, IntervalStrategy
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint
from datasets import Dataset, load_metric

from trainer.trainer_seq2seq_qa import QuestionAnsweringSeq2SeqTrainer

import fire

set_seed(42)
logger = logging.getLogger(__name__)

paths = {}
paths["QA_BERT_lastlayer_binarychoice"] = "./socialiq/SOCIAL-IQ_QA_BERT_LASTLAYER_BINARY_CHOICE.csd"
paths["DENSENET161_1FPS"] = "./deployed/SOCIAL_IQ_DENSENET161_1FPS.csd"
paths["Transcript_Raw_Chunks_BERT"] = "./deployed/SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT.csd"
paths["Acoustic"] = "./deployed/SOCIAL_IQ_COVAREP.csd"
social_iq = mmdatasdk.mmdataset(paths)
social_iq.unify()


def flatten_qail(_input):
    return _input.reshape(-1,*(_input.shape[3:])).squeeze().transpose(1,0,2)


def to_pytorch(_input):
    return Variable(torch.tensor(_input)).cuda()


def reshape_to_correct(_input,shape):
    return _input[:,None,None,:].expand(-1,shape[1],shape[2],-1).reshape(-1,_input.shape[1])


def get_judge(load_data=None):
    if load_data=='a':
        input_dim=1024
    elif load_data=='qa':
        input_dim=1024*2
    elif load_data=='qat':
        input_dim=120
    elif load_data=='qav':
        input_dim=150
    else:
        input_dim=340
    return nn.Sequential(OrderedDict([
        ('fc0',   nn.Linear(input_dim,25)),
        ('sig0', nn.Sigmoid()),
        ('fc1',   nn.Linear(25,1)),
        ('sig1', nn.Sigmoid())
        ]))


def calc_accuracy(correct,incorrect):
    correct_=correct.cpu()
    incorrect_=incorrect.cpu()
    return numpy.array(correct_>incorrect_,dtype="float32").sum()/correct.shape[0]


def qai_to_tensor(in_put,keys,total_i=1):
    data=dict(in_put.data)
    features=[]
    for i in range (len(keys)):
        features.append(numpy.array(data[keys[i]]["features"]))
    input_tensor=numpy.array(features,dtype="float32")[:,0,...]
    in_shape=list(input_tensor.shape)
    q_tensor=input_tensor[:,:,:,0:1,:,:]
    ai_tensor=input_tensor[:,:,:,1:,:,:]
    return q_tensor,ai_tensor[:,:,:,0:1,:,:],ai_tensor[:,:,:,1:1+total_i,:,:]


def build_qa_binary(qa_glove, keys):
    return qai_to_tensor(qa_glove, keys, 1)


def build_visual(visual, keys):
    vis_features = []
    for i in range(len(keys)):
        this_vis = numpy.array(visual[keys[i]]["features"])
        numpy.nan_to_num(this_vis)
        this_vis = numpy.concatenate([this_vis, numpy.zeros([25, 2208])], axis=0)[:25, :]
        vis_features.append(this_vis)
    return numpy.array(vis_features, dtype="float32").transpose(1, 0, 2)


def build_acc(acoustic, keys):
    acc_features = []
    for i in range(len(keys)):
        this_acc = numpy.array(acoustic[keys[i]]["features"])
        numpy.nan_to_num(this_acc)
        this_acc = numpy.concatenate([this_acc, numpy.zeros([25, 74])], axis=0)[:25, :]
        acc_features.append(this_acc)
    final = numpy.array(acc_features, dtype="float32").transpose(1, 0, 2)
    return numpy.array(final, dtype="float32")


def build_trs(trs, keys):
    trs_features = []
    for i in range(len(keys)):
        this_trs = numpy.array(trs[keys[i]]["features"][:, -768:])
        numpy.nan_to_num(this_trs)
        this_trs = numpy.concatenate([this_trs, numpy.zeros([25, 768])], axis=0)[:25, :]
        trs_features.append(this_trs)
    return numpy.array(trs_features, dtype="float32").transpose(1, 0, 2)


def process_data(keys):
    qa_glove = social_iq["QA_BERT_lastlayer_binarychoice"]
    visual = social_iq["DENSENET161_1FPS"]
    transcript = social_iq["Transcript_Raw_Chunks_BERT"]
    acoustic = social_iq["Acoustic"]

    qas = build_qa_binary(qa_glove, keys)
    visual = build_visual(visual, keys)
    trs = build_trs(transcript, keys)
    acc = build_acc(acoustic, keys)

    return qas, visual, trs, acc


def feed_forward(keys, t5_model, fc, q_lstm, a_lstm, v_lstm, t_lstm, ac_lstm, mfn_mem, mfn_delta1, mfn_delta2, mfn_tfn,
                 preloaded_data=None):
    q, a, i = [data[keys[0]:keys[1]] for data in preloaded_data[0]]
    vis = preloaded_data[1][:, keys[0]:keys[1], :]
    trs = preloaded_data[2][:, keys[0]:keys[1], :]
    acc = preloaded_data[3][:, keys[0]:keys[1], :]

    reference_shape = q.shape

    q_h = t5_model(inputs_embeds=fc(to_pytorch(flatten_qail(q)))).last_hidden_state
    q_rep = torch.cat((q_h[0, :, :], q_h[-1, :, :]), dim=-1)
    a_h = t5_model(inputs_embeds=fc(to_pytorch(flatten_qail(a)))).last_hidden_state
    a_rep = torch.cat((a_h[0, :, :], a_h[-1, :, :]), dim=-1)
    i_h = t5_model(inputs_embeds=fc(to_pytorch(flatten_qail(i)))).last_hidden_state
    i_rep = torch.cat((i_h[0, :, :], i_h[-1, :, :]), dim=-1)

    # # transcript representation
    # t_full = t_lstm.step(to_pytorch(trs))
    # # visual representation
    # v_full = v_lstm.step(to_pytorch(vis))
    # # acoustic representation
    # ac_full = ac_lstm.step(to_pytorch(acc))
    #
    # t_seq = t_full[0]
    # v_seq = v_full[0]
    # ac_seq = ac_full[0]
    #
    # t_rep_extended = reshape_to_correct(t_full[1][0][0, :, :], reference_shape)
    # v_rep_extended = reshape_to_correct(v_full[1][0][0, :, :], reference_shape)
    # ac_rep_extended = reshape_to_correct(ac_full[1][0][0, :, :], reference_shape)
    #
    # # MFN and TFN Dance!
    # before_tfn = torch.cat([mfn_delta2((mfn_delta1(
    #     torch.cat([t_seq[i], t_seq[i + 1], v_seq[i], v_seq[i + 1], ac_seq[i], ac_seq[i + 1]], dim=1)) * torch.cat(
    #     [t_seq[i], t_seq[i + 1], v_seq[i], v_seq[i + 1], ac_seq[i], ac_seq[i + 1]], dim=1)))[None, :, :] for i in
    #                         range(t_seq.shape[0] - 1)], dim=0)
    # after_tfn = torch.cat(
    #     [mfn_tfn.fusion([before_tfn[i, :, :50], before_tfn[i, :, 50:70], before_tfn[i, :, 70:]])[None, :, :] for i in
    #      range(t_seq.shape[0] - 1)], dim=0)
    # after_mfn = mfn_mem.step(after_tfn)[1][0][0, :, :]
    # mfn_final = reshape_to_correct(after_mfn, reference_shape)

    return q_rep, a_rep, i_rep#, t_rep_extended, v_rep_extended, ac_rep_extended, mfn_final


def init_tensor_mfn_modules(path):
    t5_model = T5EncoderModel.from_pretrained("t5-small").cuda()
    fc = nn.Linear(768, 512).cuda()
    q_lstm = mylstm.MyLSTM(768, 50).cuda()
    a_lstm = mylstm.MyLSTM(768, 50).cuda()
    t_lstm = mylstm.MyLSTM(768, 50).cuda()
    v_lstm = mylstm.MyLSTM(2208, 20).cuda()
    ac_lstm = mylstm.MyLSTM(74, 20).cuda()

    checkpoint = torch.load(path)
    t5_model.load_state_dict(checkpoint['modelA_state_dict'])
    fc.load_state_dict(checkpoint['modelB_state_dict'])

    mfn_mem = mylstm.MyLSTM(100, 100).cuda()
    mfn_delta1 = nn.Sequential(OrderedDict([
        ('fc0', nn.Linear(180, 25)),
        ('relu0', nn.ReLU()),
        ('fc1', nn.Linear(25, 180)),
        ('relu1', nn.Softmax(dim=0))
    ])).cuda()

    mfn_delta2 = nn.Sequential(OrderedDict([
        ('fc0', nn.Linear(180, 90)),
        ('relu0', nn.ReLU()),
    ])).cuda()

    mfn_tfn = TensorFusion([50, 20, 20], 100).cuda()
    return t5_model, fc, q_lstm, a_lstm, t_lstm, v_lstm, ac_lstm, mfn_mem, mfn_delta1, mfn_delta2, mfn_tfn


if __name__ == '__main__':
    # if you have enough RAM, specify this as True - speeds things up ;)
    load_data = 'a'
    noise = 'none'
    bs = 4
    trk, dek = mmdatasdk.socialiq.standard_folds.standard_train_fold, \
               mmdatasdk.socialiq.standard_folds.standard_valid_fold
    # This video has some issues in training set
    bads = ['f5NJQiY9AuY', 'aHBLOkfJSYI']
    folds = [trk, dek]
    for bad in bads:
        for fold in folds:
            try:
                fold.remove(bad)
            except:
                pass

    t5_model, fc, q_lstm, a_lstm, t_lstm, v_lstm, ac_lstm, mfn_mem, mfn_delta1, mfn_delta2, mfn_tfn = init_tensor_mfn_modules('best.pt')

    preloaded_train = process_data(trk)
    preloaded_dev = process_data(dek)
    print("Preloading Complete")

    # Getting the Judge
    judge = get_judge(load_data).cuda()

    # Initializing parameter optimizer
    params = list(t5_model.parameters()) + list(fc.parameters()) + list(q_lstm.parameters()) + \
        list(a_lstm.parameters()) + list(judge.parameters()) + list(t_lstm.parameters()) + \
        list(v_lstm.parameters()) + list(ac_lstm.parameters()) + list(mfn_mem.parameters()) + \
        list(mfn_delta1.parameters()) + list(mfn_delta2.parameters()) + list(mfn_tfn.linear_layer.parameters())

    optimizer = optim.AdamW(params, lr=1e-4)

    best_acc = 0
    _accs = []
    ds_size = len(dek)
    if ds_size % bs == 0:
        iters = int(ds_size / bs)
    else:
        iters = int(ds_size / bs) + 1
    correct_answer_arrays = []
    incorrect_answer_arrays = []
    for j in range(iters):
        this_dek = [j * bs, (j + 1) * bs]

        q_rep, a_rep, i_rep = \
            feed_forward(this_dek, t5_model, fc, q_lstm, a_lstm, v_lstm, t_lstm,
                         ac_lstm, mfn_mem, mfn_delta1, mfn_delta2, mfn_tfn, preloaded_dev)

        real_bs = float(q_rep.shape[0])

        correct_answer_arrays.append(a_rep.cpu().detach().numpy())
        incorrect_answer_arrays.append(i_rep.cpu().detach().numpy())

        if load_data == 'a':
            correct = judge(a_rep)
            incorrect = judge(i_rep)
        if load_data == 'qa':
            correct = judge(torch.cat((q_rep, a_rep), 1))
            incorrect = judge(torch.cat((q_rep, i_rep), 1))
        # elif load_data == 'qat':
        #     correct = judge(torch.cat((q_rep, a_rep, t_rep), 1))
        #     incorrect = judge(torch.cat((q_rep, i_rep, t_rep), 1))
        # elif load_data == 'qav':
        #     correct = judge(torch.cat((q_rep, a_rep, v_rep), 1))
        #     incorrect = judge(torch.cat((q_rep, i_rep, v_rep), 1))
        # elif load_data == 'qaac':
        #     correct = judge(torch.cat((q_rep, a_rep, ac_rep), 1))
        #     incorrect = judge(torch.cat((q_rep, i_rep, ac_rep), 1))
        # else:
        #     correct = judge(torch.cat((q_rep, a_rep, i_rep, t_rep, v_rep, ac_rep, mfn_rep), 1))
        #     incorrect = judge(torch.cat((q_rep, i_rep, a_rep, t_rep, v_rep, ac_rep, mfn_rep), 1))

        if noise == 'replace_incorrect_with_incorrect':
            rand_index = torch.randperm(len(incorrect))
            incorrect = incorrect[rand_index]
        elif noise == 'replace_incorrect_with_correct':
            rand_index = torch.randperm(len(correct))
            incorrect = correct[rand_index]
        elif noise == 'replace_correct_with_correct':
            rand_index = torch.randperm(len(correct))
            correct = correct[rand_index]
        elif noise == 'replace_correct_with_incorrect':
            rand_index = torch.randperm(len(incorrect))
            correct = incorrect[rand_index]

        correct_mean = Variable(torch.Tensor(numpy.array([1.0])), requires_grad=False).cuda()
        incorrect_mean = Variable(torch.Tensor(numpy.array([0.])), requires_grad=False).cuda()

        _accs.append(calc_accuracy(correct, incorrect))

    print("Dev Accs %f", numpy.array(_accs, dtype="float32").mean())
    print("-----------")
    acc_temp = numpy.array(_accs, dtype="float32").mean()
    if acc_temp > best_acc:
        best_acc = acc_temp
    print(best_acc)

    answer_arrays = numpy.concatenate(
        (correct_answer_arrays + incorrect_answer_arrays), axis=0
    )
    label_arrays = numpy.concatenate(
        (numpy.ones(numpy.concatenate(correct_answer_arrays).shape[0]),
         numpy.zeros(numpy.concatenate(incorrect_answer_arrays).shape[0]))
    )
    print(answer_arrays.shape, label_arrays.shape)
    reducted_answer_arrays = dim_reduction(answer_arrays)
    print(reducted_answer_arrays.shape)
    visualization(reducted_answer_arrays, label_arrays)

