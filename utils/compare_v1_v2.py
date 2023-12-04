import mmsdk
from mmsdk import mmdatasdk

import os
import json

import pandas as pd

trk_v1, dek_v1 = mmdatasdk.socialiq.standard_folds.standard_train_fold, \
        mmdatasdk.socialiq.standard_folds.standard_valid_fold

directory_v1 = "/Users/xguo0038/Documents/Workspace/CCU-subteam-1/data/raw/qa"
directory_v2 = "/Users/xguo0038/Documents/Workspace/Social-IQ-2.0-Challenge/siq2/qa"


def intersection(list1, list2):
    list3 = [value for value in list1 if value in list2]
    return list3


def get_video_key(directory, split):
    with open(os.path.join(directory, 'qa_' + split + '.json'), 'r') as f:
        json_data = [json.loads(line) for line in f]
    video_keys = []
    for item in json_data:
        current_video_key = item['qid'].split('_')[0]
        if current_video_key not in video_keys:
            video_keys.append(current_video_key)
    return video_keys


trk_v2 = get_video_key(directory_v2, 'train')
dek_v2 = get_video_key(directory_v2, 'val')

# print(len(intersection(trk, trk_v2)))
# print(len(intersection(dek, dek_v2)))
print(len(trk_v1+dek_v1), len(trk_v2+dek_v2))
print(len(intersection(trk_v1+dek_v1, trk_v2+dek_v2)))


def get_question_answer_v1(directory, video_keys):
    collection_v1 = {}
    for video_key in video_keys:
        question_list, correct_answer_list, incorrect_answer_list = [], [], []
        with open(os.path.join(directory, video_key+'_trimmed.txt'), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                split_line = line.strip().split(': ')
                sign, text = split_line[0], ': '.join(split_line[1:])
                if sign[0] == 'q':
                    question_list.append(text)
                elif sign[0] == 'a':
                    correct_answer_list.append(text)
                elif sign[0] == 'i':
                    incorrect_answer_list.append(text)
        collection_v1.update({video_key:{"question": question_list,
                                         "correct_answers": correct_answer_list,
                                         "incorrect_answers": incorrect_answer_list}})
    return collection_v1


def get_question_answer_v2(directory, video_keys):
    json_data = []
    for split in ['train', 'val']:
        with open(os.path.join(directory, 'qa_' + split + '.json'), 'r') as f:
            data = [json.loads(line) for line in f]
            json_data.extend(data)

    collection_v2 = {}
    question_list, correct_answer_list, incorrect_answer_list = [], [], []
    previous_video_key, current_video_key = None, None
    for item in json_data:
        current_video_key = item['qid'].split('_')[0]
        if previous_video_key is not None:
            if current_video_key != previous_video_key:
                collection_v2.update({current_video_key: {"question": question_list,
                                                          "correct_answers": correct_answer_list,
                                                          "incorrect_answers": incorrect_answer_list}})
                question_list, correct_answer_list, incorrect_answer_list = [], [], []
        if current_video_key in video_keys:
            question_list.append(item['q'])
            answer_index = item['answer_idx']
            for i in range(4):
                if i == answer_index:
                    correct_answer_list.append(item['a'+str(i)])
                else:
                    incorrect_answer_list.append(item['a'+str(i)])
        previous_video_key = current_video_key
    collection_v2.update({current_video_key: {"question": question_list,
                                              "correct_answers": correct_answer_list,
                                              "incorrect_answers": incorrect_answer_list}})
    return collection_v2


inter_video_keys = intersection(trk_v1+dek_v1, trk_v2+dek_v2)
question_answer_v1 = get_question_answer_v1(directory_v1, inter_video_keys)
question_answer_v2 = get_question_answer_v2(directory_v2, inter_video_keys)

excel_data = []
for inter_video_key in inter_video_keys:
    question_v1 = question_answer_v1[inter_video_key]['question']
    question_v2 = question_answer_v2[inter_video_key]['question']
    inter_questions = intersection(question_v1, question_v2)

    correct_answer_v1 = question_answer_v1[inter_video_key]['correct_answers']
    correct_answer_v2 = question_answer_v2[inter_video_key]['correct_answers']
    inter_correct_answers = intersection(correct_answer_v1, correct_answer_v2)

    incorrect_answer_v1 = question_answer_v1[inter_video_key]['incorrect_answers']
    incorrect_answer_v2 = question_answer_v2[inter_video_key]['incorrect_answers']
    inter_incorrect_answers = intersection(incorrect_answer_v1, incorrect_answer_v2)

    excel_data.append([inter_video_key,
                       len(inter_questions), len(question_v1), len(question_v2),
                       len(inter_correct_answers), len(correct_answer_v1), len(correct_answer_v2),
                       len(inter_incorrect_answers), len(incorrect_answer_v1), len(incorrect_answer_v2)])

df = pd.DataFrame(excel_data, columns=['vid',
                                       'inter_q_len', 'v1_q_len', 'v2_q_len',
                                       'inter_a_len', 'v1_a_len', 'v2_a_len',
                                       'inter_i_len', 'v1_i_len', 'v2_i_len'])
df.to_excel('saved_excel.xlsx', index=False)