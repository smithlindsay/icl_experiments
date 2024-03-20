import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

class AugmentedData(torch.utils.data.Dataset):
    def __init__(self, base_dataset, num_tasks=32,num_classes=10,include_identity=False):
        self.base_dataset = base_dataset
        self.num_tasks = num_tasks
        self.image_shape = base_dataset[0][0].squeeze().shape
        self.nx = self.image_shape[0] * self.image_shape[1]
        self.C = num_classes
        self.base_len = len(base_dataset)
        self.include_identity = include_identity

        t, p = self.create_transforms()
        self.transforms = t
        self.perms = p

    def create_transforms(self):
        transforms = []
        perms = []
        for i in range(self.num_tasks):
            a = torch.normal(0, 1/self.nx, size=(self.nx, self.nx))
            p = torch.randperm(self.C)
            transforms.append(a)
            perms.append(p)
        if self.include_identity:
            transforms[0] = torch.eye(self.nx)
            perms[0] = torch.arange(0,self.C)
        return transforms, perms

    def get_transforms(self):
        return self.transforms, self.perms

    def adopt_transforms(self, other):
        assert other.num_tasks == self.num_tasks

        t, p = other.get_transforms()
        self.transforms = t
        self.perms = p

    def __getitem__(self, index):
        base_idx = index % self.base_len
        task_idx = index // self.base_len

        base_img, base_target = self.base_dataset.__getitem__(base_idx)
        transform, perm = self.transforms[task_idx], self.perms[task_idx]

        target = perm[base_target]
        base_img = base_img.view(-1, self.nx).squeeze()
        transformed = transform @ base_img
        img = transformed.view(self.image_shape)

        return img, target

    def __len__(self):
        return self.base_len*self.num_tasks