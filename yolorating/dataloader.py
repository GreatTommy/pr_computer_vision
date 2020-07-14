"""This module defines the functions used to load data and construct datasets for the
linear regression and the trace rating neural network.

"""

import os
import random

import numpy as np

from .ratings import _get_rating_local


def _create_input_feature_reg(path, label, confidence_threshold):
    f = open(os.path.join(path, label), "r")
    lines = f.readlines()
    f.close()

    inputs = [0] * 4  # [nbBlobs, nbDots, totalAreaBlobs, totalAreaDots]
    for line in lines:
        splited_line = line.split(" ")
        if float(splited_line[5]) > confidence_threshold:
            temp_inputs = [0] * 4
            dec = 0  # shift blob
            if int(splited_line[0]):  # shift dot
                dec = 1
            temp_inputs[0 + dec] += 1
            temp_inputs[2 + dec] += float(splited_line[3]) * float(splited_line[4])
            for i, _ in enumerate(inputs):
                inputs[i] += temp_inputs[i]
    return inputs


def _create_input_feature_nn(path, label, confidence_threshold):
    f = open(os.path.join(path, label), "r")
    lines = f.readlines()
    f.close()

    inputs = [0] * 30
    counter = 0
    for line in lines:
        splited_line = line.split(" ")
        if float(splited_line[5]) > confidence_threshold and counter < 10:
            inputs[counter * 3 + 0] = float(splited_line[0])  # class
            inputs[counter * 3 + 1] = float(splited_line[3]) * float(
                splited_line[4]
            )  # defect area
            inputs[counter * 3 + 2] = float(splited_line[5])  # defect confidence
            counter += 1
    return inputs


def _create_dataset_regression(path):
    labels = os.listdir(path)
    inputs = []
    outputs = []
    for label in labels:
        splited_label = label.split("_")
        inputs.append(_create_input_feature_reg(path, label, 0.01))
        image_id = int(splited_label[0])
        trace = int(splited_label[1].split(".")[0])
        outputs.append(_get_rating_local(image_id, trace))
    return (np.array(inputs), np.array(outputs))


def _construct_indices(path, dataset, train_mode=False, ratio_val=0.2):
    random.seed(0)

    indices = []
    indices_z = []
    for idx, file in enumerate(dataset.labels):
        if os.path.getsize(os.path.join(path, file)) == 0:
            indices_z.append(idx)
        else:
            indices.append(idx)
    random.shuffle(indices)
    random.shuffle(indices_z)
    if train_mode:
        sep = int(ratio_val * len(indices))
        sep_z = int(ratio_val * len(indices_z))
        return (
            indices[sep:],
            indices[:sep],
            indices_z[sep_z:],
            indices_z[:sep_z],
        )
    else:
        return (indices, indices_z)
