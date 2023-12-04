import logging
from utils import constants
import os
os.environ['TRANSFORMERS_CACHE'] = constants.TRANSFORMERS_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = constants.DATASETS_CACHE_DIR

import numpy
from mmsdk import mmdatasdk
from mmsdk.mmmodelsdk.fusion.tensor_fusion.model import TensorFusion as TensorFusion
import socialiq.mylstm as mylstm

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from transformers import set_seed
from transformers import T5EncoderModel, T5Config

import random

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
    return _input.reshape(-1, *(_input.shape[3:])).squeeze()


def to_pytorch(_input):
    return Variable(torch.tensor(_input)).cuda()


def reshape_to_correct(_input, shape):
    return _input[:, None, None, :].expand(-1, shape[1], shape[2], -1).reshape(-1, _input.shape[1])


def reshape_to_correct_context(_input, shape):
    return _input[:, None, None, :, :].expand(-1, shape[1], shape[2], -1, -1).reshape(-1, _input.shape[1], _input.shape[2])


def get_judge(load_data=None, context_matter=False):
    if context_matter:
        input_dim = 512
    elif load_data == 'a':
        input_dim = 1024
    elif load_data == 'ai':
        input_dim = 1024*2
    elif load_data == 'qa':
        input_dim = 1024*2
    elif load_data == 'qai':
        input_dim = 1024*3
    elif load_data == 'qat':
        input_dim = 1024*3
    elif load_data == 'qait':
        input_dim = 1024*4
    elif load_data == 'qav':
        input_dim = 1024*3
    elif load_data == 'qaiv':
        input_dim = 1024*4
    else:
        input_dim = 340
    return nn.Sequential(OrderedDict([
        ('fc0',   nn.Linear(input_dim, 25)),
        ('sig0', nn.Sigmoid()),
        ('fc1',   nn.Linear(25, 1)),
        ('sig1', nn.Sigmoid())
        ]))


def calc_accuracy(correct, incorrect):
    correct_ = correct.cpu()
    incorrect_ = incorrect.cpu()
    return numpy.array(correct_ > incorrect_, dtype="float32").sum()/correct.shape[0]


def qai_to_tensor(in_put, keys, total_i=1):
    data = dict(in_put.data)
    features = []
    for i in range(len(keys)):
        features.append(numpy.array(data[keys[i]]["features"]))
    input_tensor = numpy.array(features,dtype="float32")[:, 0, ...]
    in_shape = list(input_tensor.shape)
    q_tensor = input_tensor[:, :, :, 0:1, :, :]
    ai_tensor = input_tensor[:, :, :, 1:, :, :]
    return q_tensor, ai_tensor[:, :, :, 0:1, :, :], ai_tensor[:, :, :, 1:1+total_i, :, :]


def build_qa_binary(qa_glove, keys):
    return qai_to_tensor(qa_glove, keys, 1)


def build_visual(visual, keys):
    vis_features = []
    for i in range(len(keys)):
        this_vis = numpy.array(visual[keys[i]]["features"])
        numpy.nan_to_num(this_vis)
        this_vis = numpy.concatenate([this_vis, numpy.zeros([25, 2208])], axis=0)[:25, :]
        vis_features.append(this_vis)
    return numpy.array(vis_features, dtype="float32")


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
    return numpy.array(trs_features, dtype="float32")


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
                 preloaded_data=None, perturbed_data=None, v_fc=None, context_matters=False, load_data=None, offset=None):
    if perturbed_data is not None:
        q, a, _ = [data[keys[0]:keys[1]] for data in preloaded_data[0]]
        if offset is not None and offset > 0:
            _, i, _ = [data[keys[0]:keys[1]] for data in perturbed_data[0]]
            i = numpy.concatenate((i[:, :, 0, :, :, :], i[:, :, 3, :, :, :],
                                   i[:, :, 6, :, :, :], i[:, :, 4, :, :, :],
                                   i[:, :, 7, :, :, :], i[:, :, 9, :, :, :],
                                   i[:, :, 8, :, :, :], i[:, :, 10, :, :, :],
                                   i[:, :, 1, :, :, :], i[:, :, 11, :, :, :],
                                   i[:, :, 2, :, :, :], i[:, :, 5, :, :, :]),
                                  axis=2)
        elif offset is not None and offset == 0:
            a_shuffle = numpy.concatenate((a[:, 1:, :, :, :, :],
                                           numpy.expand_dims(a[:, 0, :, :, :, :], axis=1)), axis=1)
            i = numpy.concatenate((a_shuffle[:, :, 3, :, :, :], a_shuffle[:, :, 6, :, :, :], a_shuffle[:, :, 9, :, :, :],
                                   a_shuffle[:, :, 0, :, :, :], a_shuffle[:, :, 6, :, :, :], a_shuffle[:, :, 9, :, :, :],
                                   a_shuffle[:, :, 0, :, :, :], a_shuffle[:, :, 3, :, :, :], a_shuffle[:, :, 9, :, :, :],
                                   a_shuffle[:, :, 0, :, :, :], a_shuffle[:, :, 3, :, :, :], a_shuffle[:, :, 6, :, :, :]),
                                  axis=2)
            i = numpy.expand_dims(i, axis=3)
        # print(i.shape)
    else:
        q, a, i = [data[keys[0]:keys[1]] for data in preloaded_data[0]]
    trs = preloaded_data[2][keys[0]:keys[1], :, :]
    vis = preloaded_data[1][keys[0]:keys[1], :, :]
    acc = preloaded_data[3][keys[0]:keys[1], :, :]
    # print(q.shape, preloaded_data[2].shape, trs.shape)
    # q.shape: (bs, 6, 12, 1, 25, 768)
    # preloaded_data[2].shape: (888, 25, 768)
    # trs.shape: (bs, 25, 768)

    reference_shape = q.shape
    q_rep, a_rep, i_rep, t_rep, v_rep = None, None, None, None, None
    q_e = to_pytorch(flatten_qail(q))
    a_e = to_pytorch(flatten_qail(a))
    i_e = to_pytorch(flatten_qail(i))
    t_e = reshape_to_correct_context(to_pytorch(trs), reference_shape)
    v_e = reshape_to_correct_context(to_pytorch(vis), reference_shape)
    # print(q_e.shape, a_e.shape, i_e.shape, t_e.shape, v_e.shape)
    if context_matters:
        if load_data == 'a':
            a_h = t5_model(inputs_embeds=fc(a_e)).last_hidden_state.transpose(0, 1)
            i_h = t5_model(inputs_embeds=fc(i_e)).last_hidden_state.transpose(0, 1)
        elif load_data == 'qa':
            a_h = t5_model(inputs_embeds=fc(torch.cat((q_e, a_e), dim=1))).last_hidden_state.transpose(0, 1)
            i_h = t5_model(inputs_embeds=fc(torch.cat((q_e, i_e), dim=1))).last_hidden_state.transpose(0, 1)
        elif load_data == 'qat':
            a_h = t5_model(inputs_embeds=fc(torch.cat((q_e, a_e, t_e), dim=1))).last_hidden_state.transpose(0, 1)
            i_h = t5_model(inputs_embeds=fc(torch.cat((q_e, i_e, t_e), dim=1))).last_hidden_state.transpose(0, 1)
        elif load_data == 'qav':
            a_h = t5_model(inputs_embeds=torch.cat((fc(torch.cat((q_e, a_e), dim=1)), v_fc(v_e)),
                                                   dim=1)).last_hidden_state.transpose(0, 1)
            i_h = t5_model(inputs_embeds=torch.cat((fc(torch.cat((q_e, i_e), dim=1)), v_fc(v_e)),
                                                   dim=1)).last_hidden_state.transpose(0, 1)
        elif load_data == 'qatv':
            a_h = t5_model(inputs_embeds=torch.cat((fc(torch.cat((q_e, a_e, t_e), dim=1)), v_fc(v_e)),
                                                   dim=1)).last_hidden_state.transpose(0, 1)
            i_h = t5_model(inputs_embeds=torch.cat((fc(torch.cat((q_e, i_e, t_e), dim=1)), v_fc(v_e)),
                                                   dim=1)).last_hidden_state.transpose(0, 1)
        else:
            a_h, i_h = None, None

        # a_rep = torch.cat((a_h[0, :, :], a_h[-1, :, :]), dim=-1)
        # i_rep = torch.cat((i_h[0, :, :], i_h[-1, :, :]), dim=-1)
        a_rep = a_h[-1, :, :]
        i_rep = i_h[-1, :, :]
    else:
        q_h = t5_model(inputs_embeds=fc(q_e)).last_hidden_state.transpose(0, 1)
        q_rep = torch.cat((q_h[0, :, :], q_h[-1, :, :]), dim=-1)
        a_h = t5_model(inputs_embeds=fc(a_e)).last_hidden_state.transpose(0, 1)
        a_rep = torch.cat((a_h[0, :, :], a_h[-1, :, :]), dim=-1)
        i_h = t5_model(inputs_embeds=fc(i_e)).last_hidden_state.transpose(0, 1)
        i_rep = torch.cat((i_h[0, :, :], i_h[-1, :, :]), dim=-1)

        # transcript representation
        t_h = t5_model(inputs_embeds=fc(t_e)).last_hidden_state.transpose(0, 1)
        t_rep = torch.cat((t_h[0, :, :], t_h[-1, :, :]), dim=-1)
        # visual representation
        v_h = t5_model(inputs_embeds=v_fc(v_e)).last_hidden_state.transpose(0, 1)
        v_rep = torch.cat((v_h[0, :, :], v_h[-1, :, :]), dim=-1)
        # acoustic representation
        # ac_full = ac_lstm.step(to_pytorch(acc))
        #
        # t_seq = t_full[0]
        # v_seq = v_full[0]
        # ac_seq = ac_full[0]
        #
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

    return q_rep, a_rep, i_rep, t_rep, v_rep#, ac_rep_extended, mfn_final


def init_tensor_mfn_modules():
    # random initialize t5-small
    # t5_config = T5Config.from_pretrained("t5-small")
    # t5_model = T5EncoderModel(t5_config).cuda()
    # t5_model = T5EncoderModel.from_pretrained("t5-small").cuda()
    t5_model = T5EncoderModel.from_pretrained("out/delphi_freeform_yesno/checkpoint-150000").cuda()
    fc = nn.Linear(768, 512).cuda()
    v_fc = nn.Linear(2208, 512).cuda()
    q_lstm = mylstm.MyLSTM(768, 50).cuda()
    a_lstm = mylstm.MyLSTM(768, 50).cuda()
    t_lstm = mylstm.MyLSTM(768, 50).cuda()
    v_lstm = mylstm.MyLSTM(2208, 20).cuda()
    ac_lstm = mylstm.MyLSTM(74, 20).cuda()

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
    return t5_model, fc, v_fc, q_lstm, a_lstm, t_lstm, v_lstm, ac_lstm, mfn_mem, mfn_delta1, mfn_delta2, mfn_tfn


if __name__ == '__main__':
    # config additional features here
    config = {
        "training_data_proportion": 1,  # (0,1]
        "redistribute_data": False,
        "perturbation_on_training_answers": True,
        "perturbation_on_development_answers": True,
        "perturbation_offset": 10,
        "dev_perturbation_offset": 20,
        "context_matter": False,
    }
    # if you have enough RAM, specify this as True - speeds things up ;)
    preload = True
    load_data = 'qav'
    bs = 8
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

    t5_model, fc, v_fc, q_lstm, a_lstm, t_lstm, v_lstm, ac_lstm, \
        mfn_mem, mfn_delta1, mfn_delta2, mfn_tfn = init_tensor_mfn_modules()

    preloaded_train, preloaded_dev = None, None
    perturbed_train, perturbed_dev = None, None
    if preload is True:
        if config['redistribute_data']:
            data = trk + dek
            random.shuffle(data)
            preloaded_train = process_data(data[:888])
            preloaded_dev = process_data(data[888:])
            for item in data[888:]:
                if item not in dek:
                    print('successfully shuffled!')
                if item in data[:888]:
                    print('overlap detected!')
        else:
            preloaded_train = process_data(trk)
            preloaded_dev = process_data(dek)
        print("Preloading Complete")
        if config['perturbation_on_training_answers']:
            offset = config['perturbation_offset']
            perturbed_train = process_data(trk[offset:] + trk[0:offset])
            print("Perturbation on Training Data Complete")
        if config['perturbation_on_development_answers']:
            offset = config['dev_perturbation_offset']
            perturbed_dev = process_data(dek[offset:] + dek[0:offset])
            print("Perturbation on Development Data Complete")
    else:
        preloaded_data = None

    # Getting the Judge
    judge = get_judge(load_data, context_matter=config['context_matter']).cuda()

    # Initializing parameter optimizer
    params = list(t5_model.parameters()) + list(fc.parameters()) + list(q_lstm.parameters()) + \
        list(a_lstm.parameters()) + list(judge.parameters()) + list(t_lstm.parameters()) + \
        list(v_lstm.parameters()) + list(ac_lstm.parameters()) + list(mfn_mem.parameters()) + \
        list(mfn_delta1.parameters()) + list(mfn_delta2.parameters()) + list(mfn_tfn.linear_layer.parameters()) + \
        list(v_fc.parameters())

    optimizer = optim.AdamW(params, lr=1e-4)

    best_acc = 0
    dev_acc = []
    dev_acc_decrease_iters = 0
    for i in range(40):
        t5_model.train()
        print("Epoch %d" % i)
        losses = []
        accs = []
        ds_size = len(trk)
        if ds_size % bs == 0:
            iters = int(ds_size / bs)
        else:
            iters = int(ds_size / bs) + 1
        iters = int(iters * config['training_data_proportion'])
        for j in range(iters):
            if preload is True:
                this_trk = [j * bs, (j + 1) * bs]
            else:
                this_trk = trk[j * bs:(j + 1) * bs]

            q_rep, a_rep, i_rep, t_rep, v_rep = \
                feed_forward(this_trk, t5_model, fc,
                             q_lstm, a_lstm, v_lstm, t_lstm, ac_lstm,
                             mfn_mem, mfn_delta1, mfn_delta2, mfn_tfn,
                             preloaded_train, perturbed_train, v_fc,
                             context_matters=config['context_matter'],
                             load_data=load_data, offset=config['perturbation_offset'])

            if load_data == 'a' or config['context_matter']:
                correct = judge(a_rep)
                incorrect = judge(i_rep)
            elif load_data == 'ai':
                correct = judge(torch.cat((a_rep, i_rep), 1))
                incorrect = judge(torch.cat((i_rep, a_rep), 1))
            elif load_data == 'qa':
                correct = judge(torch.cat((q_rep, a_rep), 1))
                incorrect = judge(torch.cat((q_rep, i_rep), 1))
            elif load_data == 'qai':
                correct = judge(torch.cat((q_rep, a_rep, i_rep), 1))
                incorrect = judge(torch.cat((q_rep, i_rep, a_rep), 1))
            elif load_data == 'qat':
                correct = judge(torch.cat((q_rep, a_rep, t_rep), 1))
                incorrect = judge(torch.cat((q_rep, i_rep, t_rep), 1))
            elif load_data == 'qait':
                correct = judge(torch.cat((q_rep, a_rep, i_rep, t_rep), 1))
                incorrect = judge(torch.cat((q_rep, i_rep, a_rep, t_rep), 1))
            elif load_data == 'qav':
                correct = judge(torch.cat((q_rep, a_rep, v_rep), 1))
                incorrect = judge(torch.cat((q_rep, i_rep, v_rep), 1))
            elif load_data == 'qaiv':
                correct = judge(torch.cat((q_rep, a_rep, i_rep, v_rep), 1))
                incorrect = judge(torch.cat((q_rep, i_rep, a_rep, v_rep), 1))
            # elif load_data == 'qaac':
            #     correct = judge(torch.cat((q_rep, a_rep, ac_rep), 1))
            #     incorrect = judge(torch.cat((q_rep, i_rep, ac_rep), 1))
            # else:
            #     correct = judge(torch.cat((q_rep, a_rep, i_rep, t_rep, v_rep, ac_rep, mfn_rep), 1))
            #     incorrect = judge(torch.cat((q_rep, i_rep, a_rep, t_rep, v_rep, ac_rep, mfn_rep), 1))

            correct_mean = Variable(torch.Tensor(numpy.array([1.0])), requires_grad=False).cuda()
            incorrect_mean = Variable(torch.Tensor(numpy.array([0.])), requires_grad=False).cuda()

            optimizer.zero_grad()
            loss = (nn.MSELoss()(correct.mean(dim=0), correct_mean) + nn.MSELoss()(incorrect.mean(dim=0),
                                                                                   incorrect_mean))
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            accs.append(calc_accuracy(correct, incorrect))

        print("Loss %f", numpy.array(losses, dtype="float32").mean())
        print("Accs %f", numpy.array(accs, dtype="float32").mean())

        _accs = []
        t5_model.eval()
        ds_size = len(dek)
        if ds_size % bs == 0:
            iters = int(ds_size / bs)
        else:
            iters = int(ds_size / bs) + 1
        for j in range(iters):
            if preload is True:
                this_dek = [j * bs, (j + 1) * bs]
            else:
                this_dek = dek[j * bs:(j + 1) * bs]

            q_rep, a_rep, i_rep, t_rep, v_rep = \
                feed_forward(this_dek, t5_model, fc,
                             q_lstm, a_lstm, v_lstm, t_lstm, ac_lstm,
                             mfn_mem, mfn_delta1, mfn_delta2, mfn_tfn,
                             preloaded_dev, perturbed_dev, v_fc,
                             context_matters=config['context_matter'],
                             load_data=load_data, offset=config['dev_perturbation_offset'])

            # real_bs = float(q_rep.shape[0])

            if load_data == 'a' or config['context_matter']:
                correct = judge(a_rep)
                incorrect = judge(i_rep)
            elif load_data == 'ai':
                correct = judge(torch.cat((a_rep, i_rep), 1))
                incorrect = judge(torch.cat((i_rep, a_rep), 1))
            elif load_data == 'qa':
                correct = judge(torch.cat((q_rep, a_rep), 1))
                incorrect = judge(torch.cat((q_rep, i_rep), 1))
            elif load_data == 'qai':
                correct = judge(torch.cat((q_rep, a_rep, i_rep), 1))
                incorrect = judge(torch.cat((q_rep, i_rep, a_rep), 1))
            elif load_data == 'qat':
                correct = judge(torch.cat((q_rep, a_rep, t_rep), 1))
                incorrect = judge(torch.cat((q_rep, i_rep, t_rep), 1))
            elif load_data == 'qait':
                correct = judge(torch.cat((q_rep, a_rep, i_rep, t_rep), 1))
                incorrect = judge(torch.cat((q_rep, i_rep, a_rep, t_rep), 1))
            elif load_data == 'qav':
                correct = judge(torch.cat((q_rep, a_rep, v_rep), 1))
                incorrect = judge(torch.cat((q_rep, i_rep, v_rep), 1))
            elif load_data == 'qaiv':
                correct = judge(torch.cat((q_rep, a_rep, i_rep, v_rep), 1))
                incorrect = judge(torch.cat((q_rep, i_rep, a_rep, v_rep), 1))
            # elif load_data == 'qaac':
            #     correct = judge(torch.cat((q_rep, a_rep, ac_rep), 1))
            #     incorrect = judge(torch.cat((q_rep, i_rep, ac_rep), 1))
            # else:
            #     correct = judge(torch.cat((q_rep, a_rep, i_rep, t_rep, v_rep, ac_rep, mfn_rep), 1))
            #     incorrect = judge(torch.cat((q_rep, i_rep, a_rep, t_rep, v_rep, ac_rep, mfn_rep), 1))

            correct_mean = Variable(torch.Tensor(numpy.array([1.0])), requires_grad=False).cuda()
            incorrect_mean = Variable(torch.Tensor(numpy.array([0.])), requires_grad=False).cuda()

            _accs.append(calc_accuracy(correct, incorrect))

        print("Dev Accs %f", numpy.array(_accs, dtype="float32").mean())
        print("-----------")
        acc_temp = numpy.array(_accs, dtype="float32").mean()
        if acc_temp > best_acc:
            best_acc = acc_temp
            torch.save({
                'epoch': i,
                'modelA_state_dict': t5_model.state_dict(),
                'modelB_state_dict': fc.state_dict(),
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


