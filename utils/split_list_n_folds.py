import os
import shutil
import csv
import numpy as np
import random
import copy

random.seed(2018)

def load_csv_file(filepath):
    list_file = []
    with open(filepath, 'r') as csv_file:
        all_lines = csv.reader(csv_file)
        for line in all_lines:
            list_file.append(line)
    list_file.remove(list_file[0])
    return list_file


def test_load_csv_file():
    list_file_path = '/media/ouc/4T_B/DuAngAng/datasets/futurelab/image_scene_training_v1/image_scene_training/training-list-0511.csv'
    list_file = load_csv_file(list_file_path)
    print(len(list_file))


def write_dict_to_csv_file(filepath, fold_dict):
    with open(filepath, 'w') as csv_file:
        dict_writer = csv.writer(csv_file, delimiter=',')
        dict_writer.writerow(['FILE_ID', 'CATEGORY_ID'])
        for key in fold_dict.keys():
            for file_id in fold_dict[key]:
                dict_writer.writerow([file_id, key])


def test_write_dict_to_csv_file():
    filepath = 'demo-test-dict.csv'
    fold_dict = {'0': ['a', 'o'],
                   '1': ['d',
                         'e'],
                   '2': ['h'],
                   '3': ['k',
                         'l']}
    write_dict_to_csv_file(filepath, fold_dict)


def shuffle2(data):
    """shuffle elements in data, but not in place"""
    data2 = data[:]
    random.shuffle(data2)
    return data2


def get_original_structure(list_file):
    original_structure_dict = {}
    for file_id, category in list_file:
        if category in original_structure_dict.keys():
            original_structure_dict[category].append(file_id)
        else:
            original_structure_dict[category] = [file_id]
    return original_structure_dict


def n_folds_split(data, n):
    each_fold_data_nums = int(round(float(len(data)) / n))
    fold_data_nums_list = [each_fold_data_nums for i in range(1, n)]
    fold_data_nums_list.append(len(data)-each_fold_data_nums*(n-1))
    data = shuffle2(data)
    fold_data_list = []
    for i in range(n):
        fold_data_list.append(
            data[each_fold_data_nums*i:sum(fold_data_nums_list[:i+1])])
    # check nothing's been lost
    assert sum(len(i) for i in fold_data_list) == len(data)
    return fold_data_list


def make_folds_dict(original_struct_dict, n):
    """
    Create dictionary representing fold-1/fold-2/.../fold-n/ file structure
    """
    assert isinstance(original_struct_dict, dict)
    n_folds_dict = {'fold-{}'.format(i): {} for i in range(1, n+1)}

    for key, value in original_struct_dict.items():
        fold_data_list = n_folds_split(data=value, n=n)
        for i in range(1, n+1):
            n_folds_dict['fold-{}'.format(i)
                         ].update({key: fold_data_list[i-1]})
    return n_folds_dict


def merge_fold_dicts(*fold_dicts):
    """
    fold-i as the val set, other folds as the training set, that's why.
    """
    merged_fold_dict = copy.deepcopy(fold_dicts[0])
    for fold_dict in fold_dicts[1:]:
        if merged_fold_dict.keys() == fold_dict.keys():
            for key in merged_fold_dict.keys():
                merged_fold_dict[key] = [*merged_fold_dict[key], *fold_dict[key]]
        else:
            raise("They don't have the same keys.")
    return merged_fold_dict


def test_merge_fold_dicts():
    fold_dict_1 = {'0': ['a', 'o'],
                   '1': ['d',
                         'e'],
                   '2': ['h'],
                   '3': ['k',
                         'l']}
    fold_dict_2 = {'0': ['b'],
                   '1': ['f', 'p'],
                   '2': ['i'],
                   '3': ['m']}
    fold_dict_3 = {'0': ['c'],
                   '1': ['g'],
                   '2': ['j'],
                   '3': ['n']}
    print(merge_fold_dicts(fold_dict_1, fold_dict_2, fold_dict_3))


def validate_train_val_dict(train_dict, val_dict):
    assert train_dict.keys() == val_dict.keys()
    for key in train_dict.keys():
        for file_id in val_dict[key]:
            assert file_id not in train_dict[key]


def test():
    list_file_path = '/Users/duang/Documents/research/futurelab/utils/demo-list.csv'
    list_file = load_csv_file(list_file_path)
    original_struct_dict = get_original_structure(list_file)
    folds_num = 3
    n_folds_dict = make_folds_dict(original_struct_dict, folds_num)
    n_folds_dict_list = [n_folds_dict[key] for key in n_folds_dict.keys()]
    for i in range(folds_num):
        print("{}-th fold:".format(i+1))
        train_dict = merge_fold_dicts(
            *n_folds_dict_list[:i], *n_folds_dict_list[i+1:])
        print("train:", hex(id(train_dict)))
        print(train_dict)
        val_dict = n_folds_dict_list[i]
        print("val:", hex(id(val_dict)))
        print(val_dict)
        validate_train_val_dict(train_dict, val_dict)
        write_dict_to_csv_file('fold-{}-train.csv'.format(i+1), train_dict)
        write_dict_to_csv_file('fold-{}-val.csv'.format(i+1), val_dict)


def validate_output_csv_files(original_csv_path, train_csv_path, val_csv_path):
    original_file_list = load_csv_file(original_csv_path)
    train_file_list = load_csv_file(train_csv_path)
    val_file_list = load_csv_file(val_csv_path)
    assert len(original_file_list) == len(train_file_list) + len(val_file_list)
    for item in original_file_list:
        assert (item in train_file_list) or (item in val_file_list) 


def main():
    list_file_path = 'train_data_lists/training-list-0511.csv'

    print('Start splitting...')
    list_file = load_csv_file(list_file_path)
    original_struct_dict = get_original_structure(list_file)
    folds_num = 11
    n_folds_dict = make_folds_dict(original_struct_dict, folds_num)
    n_folds_dict_list = [n_folds_dict[key] for key in n_folds_dict.keys()]
    for i in range(folds_num):
        train_dict = merge_fold_dicts(
            *n_folds_dict_list[:i], *n_folds_dict_list[i+1:])
        val_dict = n_folds_dict_list[i]
        validate_train_val_dict(train_dict, val_dict)
        train_csv_path = 'train_data_lists/{}-fold-{}-train.csv'.format(folds_num, i+1)
        val_csv_path = 'train_data_lists/{}-fold-{}-val.csv'.format(folds_num, i+1)
        write_dict_to_csv_file(train_csv_path, train_dict)
        write_dict_to_csv_file(val_csv_path, val_dict)
    
    validate_output_csv_files(list_file_path, train_csv_path, val_csv_path)
    print('Complete!')

if __name__ == '__main__':
    main()
