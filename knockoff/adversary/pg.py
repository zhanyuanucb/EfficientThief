#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import random
import collections

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

from knockoff.adversary.agents.pg_agent import PGAgent


class PGAdversary(object):
    def __init__(self, queryset, num_each_class, agent_params):

        # init vars
        self.queryset = queryset
        #self.n_queryset = len(self.queryset)
        #num_classes = len(queryset.classes)
        #assert batch_size % num_classes == 0, "batch_size should be divisible by num_classes"
        self.num_each_class = num_each_class

        self.num_classes = len(queryset.classes)
        self.num_agent_train_steps_per_iter = agent_params["num_agent_train_steps_per_iter"]
        self.train_batch_size = agent_params['agent_train_batch_size']
        self.agent_params = agent_params

        self.agent = PGAgent(agent_params)

        self.idx_set = set(range(len(self.queryset)))
        self.idx_counter = collections.defaultdict(int)

        self._restart()

    def _restart(self):
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        self.idx_set = set(range(len(self.queryset)))
        self.transferset = []

    def init_sampling(self):
#        X = []
#        for i in range(self.num_classes):
#            X.append(self._sample_from_class(i))

        sample_target = random.randint(0, self.num_classes-1)
        actions = np.array([sample_target])
        return self._sample_from_class(sample_target), actions

    def _sample_from_class(self, target):
        labels = np.array(self.queryset.targets)
        # TODO: Optimize by caching
        target_idx = (labels==target).nonzero()[0]
        sample_idx = np.random.choice(target_idx, size=self.num_each_class, replace=False)
        for i in sample_idx:
            self.idx_counter[i] += 1
        subset = torch.utils.data.Subset(self.queryset, sample_idx)
        return torch.stack([subset[i][0] for i in range(len(subset))]) # [(img, label)]


    def sample(self, observations):
        actions = self.agent.take_action(observations)
        X = []
        for target in actions:
            X.append(self._sample_from_class(target))
        return torch.cat(X), actions

    def train_agent(self):
        for train_step in range(self.num_agent_train_steps_per_iter):
            ob_batch, ac_batch, next_ob_batch, concatenated_rews, unconcatenated_rews = self.agent.sample_from_replay_buffer(self.train_batch_size)

            self.agent.train(ob_batch, ac_batch, next_ob_batch, concatenated_rews, unconcatenated_rews)

    def add_to_replay_buffer(self, paths):
        self.agent.add_to_replay_buffer(paths)
