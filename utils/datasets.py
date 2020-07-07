import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import math
import numpy as np
import warnings
from utils.helpers import *

class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


def get_loaders(datasets_path,
                dataset,
                train_batch_size=128,
                val_batch_size=128,
                val_source='test',
                val_train_fraction=0.1,
                val_train_overlap=False,
                workers=0,
                train_infinite=False,
                val_infinite=False):
    """
    NB: val_train_fraction and val_train_overlap only used if val_source='train'
    Note that infinite=True changes the seed/order of the batches
    """
    TrainLoader = InfiniteDataLoader if train_infinite else DataLoader
    ValLoader = InfiniteDataLoader if val_infinite else DataLoader

    ## Select relevant dataset
    if dataset in ['MNIST', 'FashionMNIST']:
        mean, std = (0.1307,), (0.3081,)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        if dataset == 'MNIST':
            train_dataset = datasets.MNIST(datasets_path, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(datasets_path, train=False, download=True, transform=transform)
        elif dataset == 'FashionMNIST':
            train_dataset = datasets.FashionMNIST(datasets_path, train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(datasets_path, train=False, download=True, transform=transform)

    elif dataset == 'SVHN':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        dataset_path = os.path.join(datasets_path, 'SVHN') #Pytorch is inconsistent in folder structure
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        train_dataset = datasets.SVHN(dataset_path, split='train', download=True, transform=transform_train)
        test_dataset = datasets.SVHN(dataset_path, split='test', download=True, transform=transform_test)
        #print(len(train_dataset))

    elif dataset in ['CIFAR10', 'CIFAR100']:
        # official CIFAR10 std seems to be wrong (actual is [0.2470, 0.2435, 0.2616])
        mean = (0.4914, 0.4822, 0.4465) if dataset == 'CIFAR10' else (0.5071, 0.4867, 0.4408)
        std = (0.2023, 0.1994, 0.2010) if dataset == 'CIFAR10' else (0.2675, 0.2565, 0.2761)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        if dataset == 'CIFAR10':
            dataset_path = os.path.join(datasets_path, 'CIFAR10') #Pytorch is inconsistent in folder structure
            train_dataset = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transform_test)
        elif dataset == 'CIFAR100':
            dataset_path = os.path.join(datasets_path, 'CIFAR100')
            train_dataset = datasets.CIFAR100(dataset_path, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR100(dataset_path, train=False, download=True, transform=transform_test)

    elif dataset == 'CINIC10':
        mean, std = (0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        dataset_path = os.path.join(datasets_path, dataset)
        train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform_train)
        test_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'test'), transform=transform_test)

    elif dataset == 'ImageNet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        dataset_path = os.path.join(datasets_path, dataset)
        train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform_train)
        test_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform=transform_test)

    else:
        print(f'{dataset} is not implemented')
        raise NotImplementedError

    ## Create dataloaders
    n_train_images = len(train_dataset)
    pin_memory = True if dataset == 'ImageNet' else False

    if val_source == 'test':
        train_loader = TrainLoader(
            dataset=train_dataset, batch_size=train_batch_size,
            shuffle=True, drop_last=True, num_workers=workers, pin_memory=pin_memory)

        val_loader = ValLoader(
            dataset=test_dataset, batch_size=val_batch_size,
            shuffle=True, drop_last=True, num_workers=workers, pin_memory=pin_memory)

    elif val_source == 'train':
        all_indices = list(range(n_train_images))
        val_indices = np.random.choice(all_indices, size=int(val_train_fraction * n_train_images), replace=False)

        val_loader = ValLoader(
            dataset=train_dataset, batch_size=val_batch_size,
            sampler=SubsetRandomSampler(val_indices), drop_last=True,
            num_workers=workers, pin_memory=pin_memory)

        if val_train_overlap:
            train_loader = TrainLoader(
                dataset=train_dataset, batch_size=train_batch_size,
                shuffle=True, drop_last=True, num_workers=workers, pin_memory=pin_memory)
        else:
            train_indices = list(set(all_indices) - set(val_indices))
            train_loader = TrainLoader(
                dataset=train_dataset, batch_size=train_batch_size,
                sampler=SubsetRandomSampler(train_indices), drop_last=True,
                num_workers=workers, pin_memory=pin_memory)

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=val_batch_size,
        shuffle=True, drop_last=True, num_workers=workers, pin_memory=pin_memory) # test loader never infinite


    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_loaders('/home/paul/Datasets/Pytorch/',
                                                        'CIFAR10',
                                                        train_batch_size=500,
                                                        val_batch_size=500,
                                                        val_source='train',
                                                        val_train_fraction=0.05,
                                                        val_train_overlap=False,
                                                        workers=0,
                                                        train_infinite=False,
                                                        val_infinite=False)

    print(len(train_loader)*500)
    print(len(val_loader)*500)
    for x_val, y_val in val_loader:
        print(x_val.shape)



















