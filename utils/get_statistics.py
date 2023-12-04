import os
import json


def get_transcript_from_single_file(file):
    transcript_list = []
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line[:2] == '00':
            continue
        elif len(line.strip()) == 0:
            continue
        elif line.strip() == "WEBVTT":
            continue
        else:
            current_transcript = line.strip()
            if current_transcript not in transcript_list:
                transcript_list.append(current_transcript)
    return transcript_list


def get_transcripts_from_directory(directory, vid_list=None):
    transcripts = {}
    for filename in os.listdir(directory):
        if vid_list and filename[:-4] not in vid_list:
            continue
        f = os.path.join(directory, filename)
        transcript = get_transcript_from_single_file(f)
        transcripts.update({os.path.splitext(filename)[0]: transcript})
    return transcripts


def print_len_of_transcripts(directory):
    len_list = []
    transcripts = get_transcripts_from_directory(directory)
    for key, value in transcripts.items():
        if key == '0-HM2VCdrC0':
            print(key, value)
        whole_value = ' '.join(value)
        len_list.append(len(whole_value.split()))
    print(max(len_list), min(len_list), sum(len_list) / len(len_list))


def read_dataset(directory, split):
    vid_list = []
    with open(os.path.join(directory, 'qa_' + split + '.json'), 'r') as f:
        json_data = [json.loads(line) for line in f]
        for item in json_data:
            vid = item['vid_name']
            if vid not in vid_list:
                vid_list.append(vid)
    return len(vid_list)


if __name__ == '__main__':
    directory_qa = "siq2/qa"
    print(read_dataset(directory_qa, 'train'))
    print(read_dataset(directory_qa, 'val'))
    # directory_v2 = "/Users/xguo0038/Documents/Workspace/Social-IQ-2.0-Challenge/siq2/transcript"
    # print_len_of_transcripts(directory_v2)
