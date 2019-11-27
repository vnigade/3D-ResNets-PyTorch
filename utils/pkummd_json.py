from __future__ import print_function, division
import os
import sys
import json
import pandas as pd


def convert_csv_to_dict(csv_path, idx_to_class, subset):
    data = pd.read_csv(csv_path, delimiter=',', header=None)

    database = {}
    file_name, file_ext = os.path.splitext(os.path.basename(csv_path))
    for i in range(data.shape[0]):
        row = data.ix[i, :]
        key = file_name + '-clip-' + str(i)
        key_label = int(row[0])
        start_frame = int(row[1])
        end_frame = int(row[2])
        confidence = int(row[3])
        database[key] = {}
        database[key]['subset'] = subset
        database[key]['annotations'] = {
            'label': key_label, 'label_name': idx_to_class[key_label],
            'start_frame': start_frame,
            'end_frame': end_frame, 'confidence': confidence}

    return database


def load_labels(label_csv_path):
    labels = []

    data = pd.read_csv(label_csv_path, delimiter=',', header=None)
    file_name, file_ext = os.path.splitext(os.path.basename(label_csv_path))
    for i in range(data.shape[0]):
        # We treat each action instance as a separate clip
        labels.append(file_name + '-clip-'+str(i))
    return labels


def load_class_names(path):
    df = pd.read_excel(path)
    index_list = df['Label']
    names_list = df['Action']
    idx_to_class = {}
    for i in df.index:
        ind = int(index_list[i])
        name = str(names_list[i])
        idx_to_class[ind] = name

    return idx_to_class


def convert_pkummd_csv_to_json(label_csv_path, idx_to_class, dst_json_path):

    labels = []
    dst_data = {}
    dst_data['database'] = {}
    for i in range(1, TOTAL_VIDEOS + 1):
        file_format = '{:04d}-' + SPLIT_TYPE + '.txt'
        file_path = os.path.join(label_csv_path, file_format.format(i))
        if os.path.exists(file_path):
            labels.extend(load_labels(file_path))

            if i <= N_TRAIN_SPLIT:
                database = convert_csv_to_dict(
                    file_path, idx_to_class, 'training')
            else:
                database = convert_csv_to_dict(
                    file_path, idx_to_class, 'validation')
            dst_data['database'].update(database)

    dst_data['labels'] = labels

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    SPLIT_TYPE = 'M'  # (L, M R)
    N_TRAIN_SPLIT = 255
    TOTAL_VIDEOS = 364

    label_csv_path = sys.argv[1]
    class_label_path = sys.argv[2]
    dst_json_path = os.path.join(
        label_csv_path, 'pkummd_{}.json'.format(SPLIT_TYPE))

    idx_to_class = load_class_names(class_label_path)
    convert_pkummd_csv_to_json(label_csv_path, idx_to_class, dst_json_path)
