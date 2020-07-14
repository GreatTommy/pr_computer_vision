"""This module defines the functions to use the trace rating neural network.

The module also include classes to define the structure and components of the network.
The functions can be used to train, test and inference the network.

"""

import os
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .dataloader import _create_input_feature_nn, _construct_indices
from .ratings import (
    _scale_rating,
    _rescale_rating,
    _get_rating_local,
    _save_rating_json,
)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 15)  # input layer => hidden layer n°1
        self.fc2 = nn.Linear(15, 15)  # hidden layer n°1 => hidden layer n°2
        self.fc3 = nn.Linear(15, 15)  # hidden layer n°2 => hidden layer n°3
        self.fc4 = nn.Linear(15, 15)  # hidden layer n°2 => hidden layer n°3
        self.fc5 = nn.Linear(15, 1)  # hidden layer n°4 => output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x


class _YoloDetections(Dataset):
    def __init__(self, path, train_mode=0):
        self.train_mode = train_mode
        self.path = path
        self.labels = os.listdir(self.path)
        self.confidence_threshold = 0.01

    def __getitem__(self, index):
        image_id = int(self.labels[index].split("_")[0])
        trace = int(self.labels[index].split("_")[1].split(".")[0])
        input_features = torch.FloatTensor(
            _create_input_feature_nn(
                self.path, self.labels[index], self.confidence_threshold
            )
        )

        if self.train_mode:
            note = torch.FloatTensor(
                [_scale_rating(_get_rating_local(image_id, trace))]
            )
            return (input_features, note)

        else:
            return (input_features, image_id, trace)

    def __len__(self):
        return len(self.labels)


class _WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super(_WeightedMSELoss, self).__init__()

    def forward(self, output, y):
        weights = [1, 1, 1, 1, 0.5]
        # after fine-tuning, these weigths seem to give good results
        loss = (output - y) ** 2

        for i, value in enumerate(loss):
            loss[i] = value * weights[int(y[i])]

        return loss.sum()


def run_inference(path: str, weights: str, input_size: int = 30) -> None:
    """Function to give a rating to traces after the process of defects detection done
    by YOLO.

    The function runs inference of a pretrained neural network to make it's predictions.
    The results of the rating process are saved in "ratings.json" file.

    Parameters
    ----------
    path : str
        The path to the folder containing YOLO's outputs (one .txt file for each trace).
    weights : str
        The path to the weights of the pretrained neural network.
    input_size : int, optional
        The size of the input feature layer.

    """

    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not isinstance(weights, str):
        raise TypeError("weights must be a string")
    if not isinstance(input_size, int):
        raise TypeError("input_size must be an integer")

    if not input_size > 0:
        raise ValueError("input_size must be positive (strictly)")

    dataset = _YoloDetections(path)
    inferenceset = DataLoader(dataset)
    network = _Net()
    network.load_state_dict(torch.load(weights))
    for data in tqdm(dataset):
        input_features, image_id, trace = data
        if torch.sum(input_features) == 0:
            _save_rating_json(10.0, int(image_id), int(trace))
        else:
            with torch.no_grad():
                output = network(input_features.view(-1, input_size))
            _save_rating_json(
                _rescale_rating(np.rint(output)), int(image_id), int(trace)
            )


def run_training(
    path: str,
    input_size: int,
    batch_size: int,
    learning_rate: Union[float, int],
    ratio_val: Union[float, int],
    n_epochs: int,
    resume: bool = False,
) -> None:
    """Function to train the traces notation neural network.

    The function runs the training pipeline of the neural network. The aim of the
    network is to learn a notation function to predict the ratings given by BIC
    operators to the traces.

    The input dataset of the function is the detections mades by the YOLO defect
    detection pipeline. The ground thruth ratings are loaded from a "ratings.csv" file.

    The input dataset will be separated in two datasets : the training set and the
    validation set.

    Every 10 epochs of the neural network training, the function will test the
    performances of the network and then display several performance metrics.

    At the end of the training, the calculated weights of the network are saved under
    yolorating/weights/last.pt

    Parameters
    ----------
    path : str
        The path to the folder containing YOLO's outputs (one .txt file for each trace).
    input_size : int
        The size of the input feature layer.
    batch_size : int
        The number of traces treated in a single batch.
    learning_rate : Union[float, int]
        The value of the optimizer's learning rate.
    ratio_val : Union[float, int]
        Proportion of the dataset who will be deticated to the validation set.
    n_epochs : int
        The number of training epochs.
    resume : bool, optional
        If true, the training will resume from a previously trained weights who must be
        located in the yolorating/weights folder under the name "last.pt"

    """

    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not isinstance(input_size, int):
        raise TypeError("input_size must be an integer")
    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if not isinstance(learning_rate, (float, int)):
        raise TypeError("learning_rate must be a number")
    if not isinstance(ratio_val, (float, int)):
        raise TypeError("batch_size must be a number")
    if not isinstance(n_epochs, int):
        raise TypeError("n_epochs must be an integer")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a boolean")

    if not input_size > 0:
        raise ValueError("input_size must be positive (strictly)")
    if not batch_size > 0:
        raise ValueError("batch_size must be positive (strictly)")
    if not learning_rate > 0:
        raise ValueError("learning_rate must be positive (strictly)")
    if not 0 <= ratio_val < 1:
        raise ValueError("ratio_val must be superior or equal to 0 and inferior to 1")
    if not n_epochs >= 0:
        raise ValueError("n_epochs must be positive")

    dataset = _YoloDetections(path, True)

    # Selection of the labels which will compose the training set and the validation set
    print(f"{len(dataset)} elements in the dataset")
    (train_indices, val_indices, train_z_indices, val_z_indices,) = _construct_indices(
        path, dataset, True, ratio_val
    )

    train_sample = SubsetRandomSampler(train_indices)
    val_sample = SubsetRandomSampler(val_indices)
    train_z_sample = SubsetRandomSampler(train_z_indices)
    val_z_sample = SubsetRandomSampler(val_z_indices)

    trainset = DataLoader(dataset, batch_size=batch_size, sampler=train_sample)
    valset = DataLoader(dataset, batch_size=batch_size, sampler=val_sample)
    trainzset = DataLoader(dataset, sampler=train_z_sample)
    valzset = DataLoader(dataset, sampler=val_z_sample)

    print(f"{len(train_indices) + len(train_z_indices)} elements in the training set")
    print(f"{len(val_indices) + len(val_z_indices)} elements in the validation set")

    network = _Net()
    print("Summary of the network :")
    print(network)

    loss_function = _WeightedMSELoss()
    optimizer = optim.Adam(network.parameters(), learning_rate)

    if resume:
        network.load_state_dict(
            torch.load(f"{os.path.dirname(__file__)}/weights/last.pt")
        )
        _run_test(input_size, trainset, valset, trainzset, valzset, network)

    # Training of the NN
    network.zero_grad()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for data in trainset:
            x, y = data
            network.zero_grad()
            output = network(x.view(-1, input_size))
            loss = loss_function(output, y)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
        print(f"Epoch n°{epoch} : Total loss => {epoch_loss}")
        if epoch % 10 == 9 and epoch > 0 or epoch == n_epochs - 1:
            _run_test(input_size, trainset, valset, trainzset, valzset, network)

    torch.save(network.state_dict(), f"{os.path.dirname(__file__)}/weights/last.pt")


def _run_test(input_size, trainset, valset, trainzset, valzset, network):
    for key, dataset in {
        "training set": (trainset, trainzset),
        "validation set": (valset, valzset),
    }.items():
        correct = 0
        total = 0

        confusion_matrix = np.zeros((5, 5))
        with torch.no_grad():
            for data in dataset[0]:
                x, y = data
                output = network(x.view(-1, input_size))
                for idx, i in enumerate(output):
                    i = int(np.rint(i))
                    if i < 0:  # out of range rating correction
                        i = 0
                    if i > 4:  # out of range rating correction
                        i = 4
                    confusion_matrix[i, int(y[idx])] += 1
                    if np.rint(i) == y[idx]:
                        correct += 1
                    total += 1
        for data in dataset[1]:
            _, y = data
            confusion_matrix[4, int(y)] += 1
            if y == 4:
                correct += 1
            total += 1

        print(f"\nAccuracy for the {key} => {round(correct / total, 3)}")
        print(f"MAE for the {key} => {round(_calculate_mae(confusion_matrix), 3)}")
        print(f"Confusion matrix for the {key}")
        print("Predicted => Ground truth :")
        print(pd.DataFrame(confusion_matrix).astype(int))


def _calculate_mae(confusion_matrix):
    sum_ae = 0
    sum_tot = 0
    for idx_i, line in enumerate(confusion_matrix):
        for idx_j, elt in enumerate(line):
            sum_ae += abs(idx_i - idx_j) * elt
            sum_tot += elt
    return sum_ae / sum_tot
