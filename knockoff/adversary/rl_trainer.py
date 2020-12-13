#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import sys
# sys.path.append('/Users/luyu/cs285_proj/EfficientThief/')
sys.path.append('/mydata/EfficientThief')
import argparse
import json
import collections
import os
import os.path as osp
import pickle
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
import torch.nn.functional as F

import knockoff.config as cfg
import knockoff.utils.model as model_utils
from knockoff import datasets
import knockoff.models.zoo as zoo
from knockoff.victim.blackbox import Blackbox

from knockoff.adversary.pg import PGAdversary
import knockoff.adversary.infrastructure.pytorch_util as ptu

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform


class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class ImageTensorSet(Dataset):
    """
    Data are saved as:
    List[data:torch.Tensor(), labels:torch.Tensor()]
    """
    def __init__(self, samples, transform=None, dataset="cifar"):
        self.data, self.targets = samples
        self.transform = transform
        self.mode = "RGB" if dataset != "mnist" else "L"

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img *= 255
            img = img.numpy().astype('uint8')
            if self.mode == "RGB":
                img = img.transpose([1, 2, 0])
            img = Image.fromarray(img, mode=self.mode)
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, str):
        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))


def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('--budgets', metavar='B', type=int,
                        help='Knockoffs will be trained for budget.')
    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)

    # RL arguments
    parser.add_argument('--traj_length', metavar='N', type=int, help='# Step in one trajactory', default=10)
    parser.add_argument('--num_each_class', metavar='N', type=int, help='# sample in each class', default=1)
    parser.add_argument('--n_iter', metavar='N', type=int, help='# iterations of RL training', default=10)
    parser.add_argument('--n_traj_each_iter', metavar='N', type=int, help='# trajactories / iter', default=10)
    parser.add_argument('--queryset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('--victim_model_dir', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Attacker's defense
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm', choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    ptu.init_gpu(gpu_id=params['device_id'])

    model_dir = params['model_dir']



    # ----------- Set up testset
    dataset_name = params['testdataset']
    valid_datasets = datasets.__dict__.keys()
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transform)
    #if len(testset.classes) != num_classes:
    #    raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))


    # ----------- Set up queryset
    queryset_name = params['queryset']
    valid_datasets = datasets.__dict__.keys()
    if queryset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[queryset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    queryset = datasets.__dict__[queryset_name](train=True, transform=transform)
    num_classes = len(queryset.classes)

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    blackbox = Blackbox.from_modeldir(blackbox_dir, device)

    # ----------- Set up adversary model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    adv_model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=10)
    adv_model = adv_model.to(device)

    # ----------- Initialize adversary
    num_each_class = params['num_each_class']
    agent_params = {"ac_dim": num_classes,
                    "ob_dim": len(testset.classes),
                    "n_layers": 4,
                    "size": 64,
                    "discrete":True,
                    "learning_rate":1e-4,
                    "num_agent_train_steps_per_iter":10,
                    "agent_train_batch_size":990,
                    "gamma":0.9,
                    "reward_to_go":True,
                    "nn_baseline":False,
                    "standardize_advantages":True,
                    "eps_random":-1
                    }
    adversary = PGAdversary(queryset, num_each_class, agent_params)

    # ----------- Set up transferset
    def collect_training_trajectories(length, n_traj=10):
        nonlocal avg_rewards, avg_components
        paths = []
        mean_rew = 0
        mean_cert = mean_L = mean_E = mean_div = 0
        X_paths, Y_paths = [], []
        for _ in range(n_traj):
            obs, acs, rewards, next_obs = [], [], [], []
            r_certs, r_Ls, r_Es, r_divs = [], [], [], []
            X_path, Y_path = [], []
            X, actions = adversary.init_sampling()
            X_path.append(X)
            ob = blackbox(X)
            Y_path.append(ob)
            ob = ob.numpy()

            for t in range(length-1):
                with torch.no_grad():
                    # Observe and react
                    obs.append(ob)
                    X_new, actions = adversary.sample(ob)
                    X_path.append(X_new)
                    acs.append(actions)

                    # Env gives feedback, which is a new observation
                    X_new = X_new.to(device)
                    ob = blackbox(X_new)
                    Y_path.append(ob)
                    ob = ob.cpu().numpy()
                    next_obs.append(ob)
                    Y_adv = adv_model(X_new)
                    Y_adv = F.softmax(Y_adv, dim=1).cpu().numpy()
                reward, r_cert, r_L, r_E, r_div = adversary.agent.calculate_reward(ob, 
                                                          np.concatenate(acs), 
                                                          Y_adv)
                rewards.append(reward)
                r_certs.append(r_cert)
                r_Ls.append(r_L)
                r_Es.append(r_E)
                r_divs.append(r_div)

            obs = np.concatenate(obs)
            acs = np.concatenate(acs)

            rewards = np.concatenate(rewards)
            mean_rew += np.mean(rewards)

            mean_cert += np.mean(np.concatenate(r_certs))
            mean_L += np.mean(np.concatenate(r_Ls))
            mean_E += np.mean(np.array(r_Es))
            mean_div += np.mean(np.array(r_divs))

            next_obs = np.concatenate(next_obs)
            path = {"observation":obs,
                    "action":acs,
                    "reward":rewards,
                    "next_observation":next_obs}
            paths.append(path)
            X_paths.append(torch.cat(X_path))
            Y_paths.append(torch.cat(Y_path))
        
        print(f"==> Avg reward: {mean_rew / n_traj}")
        avg_rewards.append(mean_rew / n_traj)
        avg_components["avg_cert"].append(mean_cert)
        avg_components["avg_L"].append(mean_L)
        avg_components["avg_E"].append(mean_E)
        avg_components["avg_div"].append(mean_div)
        return torch.cat(X_paths), torch.cat(Y_paths), paths


    traj_length = params['traj_length']
    num_each_class = params['num_each_class']
    n_iter = params['n_iter']
    X, Y = None, None
    budgets = params['budgets']
    n_traj = params['n_traj_each_iter']
    criterion_train = model_utils.soft_cross_entropy
    if traj_length > 0:
        n_iter = budgets // (traj_length*n_traj)

    print(f"==> Budget = {n_iter} x {traj_length} x {n_traj}")
    best_test_acc = []
    best_acc = -1
    avg_rewards = []
    avg_components = collections.defaultdict(list)
    for iter in range(1, n_iter+1):
        # n_iter * traj_length = budget
        print(f"==> Iteration: {iter}/{n_iter}")
        X_path, Y_path, paths = collect_training_trajectories(traj_length, n_traj=n_traj)

        adversary.add_to_replay_buffer(paths)

        adversary.train_agent()

        if X is None:
            X, Y = X_path, Y_path
        else:
            X = torch.cat((X, X_path))
            Y = torch.cat((Y, Y_path))

        transferset = ImageTensorSet((X, Y))

        # ----------- Train
        #np.random.seed(cfg.DEFAULT_SEED)
        #torch.manual_seed(cfg.DEFAULT_SEED)
        #torch.cuda.manual_seed(cfg.DEFAULT_SEED)
        optimizer = get_optimizer(adv_model.parameters(), params['optimizer_choice'], **params)
        print(f"Train on {len(transferset)} samples")
        checkpoint_suffix = '.{extraction}'
        best_acc = model_utils.train_model(adv_model, transferset, model_dir, testset=testset, criterion_train=criterion_train,
                                checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer, benchmark=best_acc, **params)
        best_test_acc.append(best_acc)
        adversary.agent.actor.save(osp.join(model_dir, "checkpoint.agent.state_dict"))

        # ----------- Log
        torch.save(best_test_acc, osp.join(model_dir, "best_acc.pylist"))
        torch.save(avg_rewards, osp.join(model_dir, "avg_rewards.pylist"))
        torch.save(avg_components, osp.join(model_dir, "avg_components.pydict"))
        torch.save(adversary.idx_counter, osp.join(model_dir, "idx_counter.pydict"))
        torch.save(transferset, osp.join(model_dir, "transferset.pt"))


    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(model_dir, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

    agent_params_out_path = osp.join(model_dir, 'agent_params_train.json')
    with open(agent_params_out_path, 'w') as jf:
        json.dump(agent_params, jf, indent=True)


if __name__ == '__main__':
    main()
