import os
import json
import torch


def check_generate_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_file_existance(file):
    if os.path.isfile(file):
        return True
    return False


def save_json(file_path, data):
    with open(file_path, 'w') as out_file:
        json.dump(data, out_file)


def read_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def save_torch_checkpoint(file_path, checkpoint):
    try:
        print('try save checkpoint')
        torch.save(checkpoint, file_path)
        return True
    except:
        print('checkpoint save error')
        return False


def load_torch_checkpoint(file_path):
    try:
        print('try load checkpoint')
        checkpoint = torch.load(file_path)
    except:
        print('load checkpoint error')
        return False
    return checkpoint
