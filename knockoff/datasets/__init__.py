from torchvision import transforms

from knockoff.datasets.caltech256 import Caltech256
from knockoff.datasets.cifarlike import CIFAR10, CIFAR100, SVHN
from knockoff.datasets.cubs200 import CUBS200
from knockoff.datasets.diabetic5 import Diabetic5
from knockoff.datasets.imagenet1k import ImageNet1k
from knockoff.datasets.indoor67 import Indoor67
from knockoff.datasets.mnistlike import MNIST, KMNIST, EMNIST, EMNISTLetters, FashionMNIST
from knockoff.datasets.tinyimagenet200 import TinyImageNet200
from knockoff.datasets.cinic10 import CINIC10
from knockoff.datasets.rlquery import RLQuery

# Create a mapping of dataset -> dataset_type
# This is helpful to determine which (a) family of model needs to be loaded e.g., imagenet and
# (b) input transform to apply
dataset_to_modelfamily = {
    # MNIST
    'MNIST': 'mnist',
    'KMNIST': 'mnist',
    'EMNIST': 'mnist',
    'EMNISTLetters': 'mnist',
    'FashionMNIST': 'mnist',

    # Cifar
    'CIFAR10': 'cifar',
    'CIFAR100': 'cifar',
    'SVHN': 'cifar',
    'TinyImageNet200': 'cifar',
    'RLQuery':'cifar',

    # Imagenet
    'CUBS200': 'imagenet',
    'Caltech256': 'imagenet',
    'Indoor67': 'imagenet',
    'Diabetic5': 'imagenet',
    'ImageNet1k': 'imagenet',

    # CINIC10
    'CINIC10': "cinic10"
}

# Transforms
modelfamily_to_transforms = {
    'mnist': {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
    },

    'cifar': {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010)),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010)),
        ]),
        'itest': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010)),
        ])

    },

    'imagenet': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    },

    'cinic10': {
        'train': transforms.Compose([
            transforms.RandomChoice([transforms.RandomHorizontalFlip(),
                                     transforms.RandomResizedCrop(32),
                                     transforms.RandomRotation(45),
                                     transforms.RandomAffine(0, translate=(0.45, 0.45)),
                                     transforms.ColorJitter(brightness=0.5),
                                     transforms.ColorJitter(contrast=0.55)
                                     ]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                 std=[0.24205776, 0.23828046, 0.25874835])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                 std=[0.24205776, 0.23828046, 0.25874835])
        ])
    }
}
