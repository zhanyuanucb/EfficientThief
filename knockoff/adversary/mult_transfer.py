#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import pickle
import json
from datetime import datetime
import sys
sys.path.append('/mydata/model-extraction/knockoffnets')
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
from knockoff.victim.blackbox import Blackbox
import knockoff.config as cfg

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class RandomAdversary(object):
    def __init__(self, blackbox, queryset, batch_size=8):
        self.blackbox = blackbox
        self.queryset = queryset

        self.n_queryset = len(self.queryset)
        self.batch_size = batch_size

        self.transferset = []  # List of tuples [(img_path, output_probs)]

        self._restart()

    def _restart(self):
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        self.transferset = []

    def get_transferset_step(self, budget, qset):
        start_B = 0
        end_B = budget
        idx_set = set(range(len(qset)))
        transferset = []
        with tqdm(total=budget) as pbar:
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):
                try:
                    idxs = np.random.choice(list(idx_set), replace=False,
                                        size=min(self.batch_size, budget - len(transferset)))
                except ValueError:
                    print(len(list(idx_set)), min(self.batch_size, budget - len(transferset)))
                    exit(1)
                #idxs = np.random.choice(list(self.idx_set), replace=False,
                #                        size=min(self.batch_size, budget - len(self.transferset)))
                idx_set = idx_set - set(idxs)

                if len(idx_set) == 0:
                    print('=> Query set exhausted. Now repeating input examples.')
                    idx_set = set(range(len(qset)))

                try:
                    x_t = torch.stack([qset[i][0] for i in idxs]).to(self.blackbox.device)
                except RuntimeError as err:
                    print(len(list(idx_set)), min(self.batch_size, budget - len(transferset)))
                    print([qset[i][0] for i in idxs])
                    print("Runtime Error: {}".format(err))
                    exit(1)
                #x_t = torch.stack([qset[i][0] for i in idxs]).to(self.blackbox.device)
                y_t = self.blackbox(x_t).cpu()

                if hasattr(qset, 'samples'):
                    # Any DatasetFolder (or subclass) has this attribute
                    # Saving image paths are space-efficient
                    img_t = [qset.samples[i][0] for i in idxs]  # Image paths
                else:
                    # Otherwise, store the image itself
                    # But, we need to store the non-transformed version
                    img_t = [qset.data[i] for i in idxs]
                    if isinstance(qset.data[0], torch.Tensor):
                        img_t = [x.numpy() for x in img_t]

                for i in range(x_t.size(0)):
                    try:
                        img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
                    except IndexError as err:
                        print(len(list(idx_set)), min(self.batch_size, budget - len(transferset)))
                        print(img_t)
                        print("Runtime Error: {}".format(err))
                        exit(1)
                    #img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
                    transferset.append((img_t_i, y_t[i].cpu().squeeze()))

                pbar.update(x_t.size(0))
        return transferset

    
    def get_transferset(self, budgets):
        for i, qset in enumerate(self.queryset):
            self.transferset.extend(self.get_transferset_step(budgets[i], qset))
        return self.transferset



def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('policy', metavar='PI', type=str, help='Policy to use while training',
                        choices=['random', 'adaptive'])
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--budgets', metavar='B', type=str,
                        help='Comma separated values of budgets. \
                        Transferset will be collected from each datasets for its corresponding budget.',
                        required=True)

    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Comma seperated list of Adversary\'s dataset (P_A(X))', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=8)
    # parser.add_argument('--topk', metavar='N', type=int, help='Use posteriors only from topk classes',
    #                     default=None)
    # parser.add_argument('--rounding', metavar='N', type=int, help='Round posteriors to these many decimals',
    #                     default=None)
    # parser.add_argument('--tau_data', metavar='N', type=float, help='Frac. of data to sample from Adv data',
    #                     default=1.0)
    # parser.add_argument('--tau_classes', metavar='N', type=float, help='Frac. of classes to sample from Adv data',
    #                     default=1.0)
    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    knockoff_utils.create_dir(out_path)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up queryset
    queryset_name = [qset for qset in params['queryset'].split(',')]
    valid_datasets = datasets.__dict__.keys()
    queryset = []
    for qset_name in queryset_name:
        if qset_name not in valid_datasets:
            raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
        modelfamily = datasets.dataset_to_modelfamily[qset_name]
        transform = datasets.modelfamily_to_transforms[modelfamily]['test']
        queryset.append(datasets.__dict__[qset_name](train=True, transform=transform))

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    blackbox = Blackbox.from_modeldir(blackbox_dir, device)

    # ----------- Initialize adversary
    batch_size = params['batch_size']
    nworkers = params['nworkers']
    transfer_out_path = osp.join(out_path, 'transferset.pickle')
    if params['policy'] == 'random':
        adversary = RandomAdversary(blackbox, queryset, batch_size=batch_size)
    elif params['policy'] == 'adaptive':
        raise NotImplementedError()
    else:
        raise ValueError("Unrecognized policy")

    print('=> constructing transfer set...')
    budgets = [int(b) for b in params['budgets'].split(',')]
    assert len(budgets) == len(queryset), "length of budgets should match the number of query datasets"
    transferset = adversary.get_transferset(budgets)
    with open(transfer_out_path, 'wb') as wf:
        pickle.dump(transferset, wf)
    print('=> transfer set ({} samples) written to: {}'.format(len(transferset), transfer_out_path))

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params_transfer.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
