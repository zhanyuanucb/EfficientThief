#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
from knockoff.victim.blackbox import Blackbox
import knockoff.config as cfg

from knockoff.adversary.agents import PGAgent


class PGAdversary(object):
    def __init__(self, queryset, batch_size, **agent_params):

        # init vars
        self.n_queryset = len(self.queryset)
        assert batch_size % num_classes == 0, "batch_size should be divisible by num_classes"
        self.num_each_class = batch_size // num_classes

        self.queryset = queryset
        self.num_classes = ac_dim

        self.agent = PGAgent(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate']
            )

        self.idx_set = set()

        self._restart()

    def _restart(self):
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        self.idx_set = set(range(len(self.queryset)))
        self.transferset = []

    def init_sampling(self):
        targets = queryset.targets

        X = []
        for i in range(self.num_classes):
            X.append(self._sample_from_class(i))

        return torch.cat(X)

    def _sample_from_class(self, target):
        labels = self.queryset.targets
        idx = np.random.choice(range(labels.size(0)), size=self.num_each_class, replace=False)
        # TODO: Optimize by caching
        sample_idx = (labels==target).nonzero()[idx]
        subset = torch.utils.data.Subset(self.queryset, target_idx).samples
        return torch.stack([sample[0] for sample in subset]) # [(img, label)]


    def sample(self, observations):
        actions = self.agent.take_action(observations)
        sample_from = actions.argmax()
        X = []
        for target in sample_from:
            X.append(self._sample_from_class(target))
        return torch.cat(X), actions

    def train_agent(self, observations, actions):
        self.agent.train(observations, actions)
