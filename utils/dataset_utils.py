import pandas as pd
import os


def form_outputs(class_label, text_label):
    return '<class>'+str(class_label)+'</class> <text>'+text_label+'</text>'


def preprocess_data(data, name):
    data['id'] = ''
    data['input_ids'] = ''
    data['labels'] = ''
    for index, row in data.iterrows():
        data.at[index, 'id'] = name + '_' + str(index)
        data.at[index, 'input_ids'] = row['input_sequence']
        data.at[index, 'labels'] = form_outputs(row['class_label'], row['text_label'])
    if "pattern" in data:
        data.drop(columns=['pattern'])
    return data


def read_tsv(path, name):
    train_data = preprocess_data(
        pd.read_csv(os.path.join(path, name, 'train.'+name+'.tsv'), sep='\t'), name)
    eval_data = preprocess_data(
        pd.read_csv(os.path.join(path, name, 'validation.'+name+'.tsv'), sep='\t'), name)
    test_data = preprocess_data(
        pd.read_csv(os.path.join(path, name, 'test.'+name+'.tsv'), sep='\t'), name)
    return train_data, eval_data, test_data
