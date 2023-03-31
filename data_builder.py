import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.datasets import FashionMNIST, CelebA
from autoaugment import Cutout
import torch

####################################################
# data loader                                      #
#                                                  #
####################################################

def build_data(dpath: str = None, batch_size=36, cutout=False, workers=1, auto_aug=False,
               dataset='CelebA', train_val_split=True):

    if dataset == 'FashionMNIST':
        mean = (0.2860,)
        std = (0.3530,)
    elif dataset == 'CelebA':
        #check and use
        mean = (0.2860,)
        std = (0.3530,)
    else:
        assert False, "Unknown dataset : {dataset}"
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]

    aug.append(transforms.ToTensor())
    if cutout:
        aug.append(Cutout(n_holes=1, length=16))
    test_dataset = None
    val_dataset = None


    if dataset == 'FashionMNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = FashionMNIST(root=dpath,
                                              train=True, transform=transform_train, download=True)
        val_dataset = FashionMNIST(root=dpath,
                                            train=False,
                                            transform=transform_test,
                                            download=True)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CelebA(root=dpath, train=True, download=True, transform=transform_train)
        val_dataset = CelebA(root=dpath, train=False, download=True, transform=transform_test)



    #multi-GPUs for distributed computation
    if torch.cuda.device_count() > 1:
        if train_val_split:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                num_workers=workers,
                drop_last=True,
                pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                sampler=val_sampler,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                drop_last=False,
                pin_memory=True)
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                sampler=test_sampler,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                drop_last=False,
                pin_memory=True)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                num_workers=workers,
                drop_last=True,
                pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                sampler=val_sampler,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                drop_last=False,
                pin_memory=True)
            test_loader = None
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=workers, pin_memory=True)
        if train_val_split:
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=workers, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=workers, pin_memory=True)
        else:
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=workers, pin_memory=True)
            test_loader = None
    return train_loader, val_loader, test_loader



