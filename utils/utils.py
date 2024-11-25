from __future__ import division

import os, sys, time, random
import torch
import json
import numpy as np

from config.config import RESULT_PATH, MODEL_PATH, PROJECT_PATH, UAP_PATH, NEURON_PATH, ATTRIBUTION_PATH

def get_model_path(dataset_name, network_arch, random_seed):
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    model_path = os.path.join(MODEL_PATH, "{}_{}_{}".format(dataset_name, network_arch, random_seed))
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    return model_path


def get_uap_path(uap_data, model_data, network_arch, random_seed):
    if not os.path.isdir(UAP_PATH):
        os.makedirs(UAP_PATH)
    model_path = os.path.join(UAP_PATH, "{}_{}_{}_{}".format(uap_data, model_data, network_arch, random_seed))
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    return model_path


def get_neuron_path():
    if not os.path.isdir(NEURON_PATH):
        os.makedirs(NEURON_PATH)
    return NEURON_PATH


def get_attribution_path():
    if not os.path.isdir(ATTRIBUTION_PATH):
        os.makedirs(ATTRIBUTION_PATH)
    return ATTRIBUTION_PATH


def get_neuron_name(uap_data, uap_arch, filter_data, filter_arch, random_seed):
    if not os.path.isdir(NEURON_PATH):
        os.makedirs(NEURON_PATH)
    model_path = "{}_{}_{}_{}_{}_outstanding.npy".format(uap_data, uap_arch, filter_data, filter_arch, random_seed)
    return model_path


def get_attribution_name(uap_data, uap_arch, random_seed):
    if not os.path.isdir(ATTRIBUTION_PATH):
        os.makedirs(ATTRIBUTION_PATH)
    model_path = "{}_{}_{}_attribution.npy".format(uap_data, uap_arch, random_seed)
    return model_path


def get_result_path(dataset_name, network_arch, random_seed, result_subfolder, postfix=''):
    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    ISOTIMEFORMAT='%Y-%m-%d_%X'
    t_string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    result_path = os.path.join(RESULT_PATH, result_subfolder, "{}_{}_{}_{}{}".format(t_string, dataset_name, network_arch, random_seed, postfix))
    os.makedirs(result_path)
    return result_path

def time_string():
    ISOTIMEFORMAT='%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

def time_file_str():
    ISOTIMEFORMAT='%Y-%m-%d'
    string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string + '-{}'.format(random.randint(1, 10000))

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def one_hot(class_labels, num_classes=None):
    if num_classes==None:
        return torch.zeros(len(class_labels), class_labels.max()+1).scatter_(1, class_labels.unsqueeze(1), 1.)
    else:
        return torch.zeros(len(class_labels), num_classes).scatter_(1, class_labels.unsqueeze(1), 1.)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_imagenet_dicts():
    # Imagenet class names
    idx2label = []
    cls2label = {}
    with open(os.path.join(PROJECT_PATH, "dataset_utils/imagenet_class_index.json"), "r") as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

    return idx2label, cls2label


def init_patch_square(data_shape, h_min, h_max, w_min, w_max):
    n, c, h, w = data_shape

    # get dummy image
    patch = torch.zeros(data_shape)
    mask = torch.zeros(data_shape)

    mask[:, :, h_min : h_max, w_min : w_max] = 1

    _patch = np.random.uniform(0.0, 1.0, (n, c, h_max - h_min, w_max - w_min))
    _patch = torch.from_numpy(_patch)

    patch[:, : , h_min: h_max, w_min: w_max] = _patch

    return patch, mask


def gap_normalize_and_scale(delta_im, batchSize):
    mean_arr = [0.485, 0.456, 0.406]
    stddev_arr = [0.229, 0.224, 0.225]
    mag_in = 10.0

    delta_im = delta_im + 1 # now 0..2
    delta_im = delta_im * 0.5 # now 0..1

    # normalize image color channels
    for c in range(3):
        delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]

    # threshold each channel of each image in deltaIm according to inf norm
    # do on a per image basis as the inf norm of each image could be different
    bs = batchSize
    for i in range(bs):
        # do per channel l_inf normalization
        for ci in range(3):
            l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
            mag_in_scaled_c = mag_in/(255.0*stddev_arr[ci])
            delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im