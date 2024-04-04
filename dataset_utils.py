import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

class RandomTasks:
    def __init__(self, base_dataset, num_tasks=32,num_classes=10):
        self.base_dataset = base_dataset
        self.num_tasks = num_tasks
        self.image_shape = base_dataset[0][0].squeeze().shape
        self.nx = self.image_shape[0] * self.image_shape[1]
        self.C = num_classes
        self.base_len = len(base_dataset)

    def get_transforms(self, task_idx):
        assert task_idx < self.num_tasks
        gen = torch.Generator(device='cpu')
        gen.manual_seed(task_idx)
        a = torch.normal(0,1/self.nx, size=(self.nx,self.nx), generator=gen)
        p = torch.randperm(self.C,generator=gen)
        return a, p

    def get_seq_from_task(self, task_idx, seq_len=100,add_channel=True):
        transform, perm = self.get_transforms(task_idx)
        data_idx = torch.randperm(self.base_len)[:seq_len]
        images = []
        targets = []
        for i in data_idx:
            image, target = self.base_dataset.__getitem__(i)
            images.append(image)
            targets.append(target)
        images = torch.concat(images)
        targets = torch.Tensor(targets).long()

        targets = perm[targets]
        images = images.view(-1, self.nx).squeeze()
        transformed = images @ transform
        output_shape = (seq_len,self.image_shape[0], self.image_shape[1])
        img = transformed.view(output_shape)

        if add_channel:
            img = img.unsqueeze(1)

        return img, targets

    def get_batch(self, batch_size, seq_len=100, device='cuda',num_channels=1):
        batch_img = torch.empty((batch_size, seq_len, num_channels, self.image_shape[0], self.image_shape[1]),device=device)
        batch_targets = torch.empty((batch_size, seq_len),device=device)
        for i in range(batch_size):
            task_idx = np.random.randint(self.num_tasks)
            img, target = self.get_seq_from_task(task_idx,seq_len=seq_len)
            batch_img[i] = img
            batch_targets[i] = target
        return batch_img, batch_targets.long()

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
    
# generate the random matrix using a predetermined random seed (have an option go in order 0, 1, 2, 3...?, and another to just generate random seed), apply it to the batch of image data (to each element of batch separately), save the seed, return the transformed image data and the seed number (seed num is the task num)
# takes in a batch of images, returns the transformed batch and the seed
# this fn is used in the training loop
# class AugDataset():    

def get_transformed_batch(images, labels, seed=None):
    device = images.device
    gen = torch.Generator()
    if seed == None:
        seed = gen.initial_seed()
    else:
        gen.manual_seed(seed)

    # images.shape = torch.Size([batch, 1, 28, 28])
    # labels.shape = torch.Size([batch])

    nx = images.shape[-2] * images.shape[-1]

    transform_mtx = torch.normal(0, 1/nx, size=(nx, nx), generator=gen).to(device)
    
    perm = torch.randperm(labels.shape[0], generator=gen)

    # matmul will broadcast the transform_mtx to each image in the batch
    transform_images = torch.matmul(images.view(-1, nx), transform_mtx).view(images.shape).to(device)
    perm_labels = labels[perm]

    return transform_images, perm_labels, seed
