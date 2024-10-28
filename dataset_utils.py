import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ExpandedDataset(Dataset):
    def __init__(self, base_dataset, expansion_factor=4):
        self.base_dataset = base_dataset
        self.expansion_factor = expansion_factor

    def __len__(self):
        return self.base_dataset.__len__() * self.expansion_factor
    
    def __getitem__(self,idx):
        base_idx = idx // self.expansion_factor
        return self.base_dataset.__getitem__(base_idx)

class SequenceLoader:
    def __init__(self, base_dataset, num_tasks=32,num_classes=10, seq_len=100, num_workers=2, 
                 batch_size=32, seed_offset=0, batches_per_epoch=100, device='cuda', 
                 dataset_expansion_factor=4):
        assert seq_len * batch_size <= len(base_dataset) * dataset_expansion_factor, "Batches bigger than dataset size are not supported"
        self.base_dataset = ExpandedDataset(base_dataset, dataset_expansion_factor)
        self.num_tasks = num_tasks
        self.image_shape = base_dataset[0][0].squeeze().shape
        self.nx = self.image_shape[0] * self.image_shape[1]
        self.C = num_classes
        self.base_len = len(base_dataset)
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.device = device
        self.batch_size = batch_size
        self.num_channels = 1
        self.seed_offset = seed_offset
        self.batches_per_epoch = batches_per_epoch
        self.loader = self._get_loader()
        self.iterator = iter(self.loader)
        self.current_iteration = 0

    def _get_transforms(self, task_idx):
        assert task_idx < self.num_tasks
        gen = torch.Generator(device='cpu')
        gen.manual_seed(task_idx + self.seed_offset)
        a = torch.normal(0,1/self.nx, size=(self.nx,self.nx), generator=gen)
        p = torch.randperm(self.C,generator=gen)
        return a, p

    def _apply_transforms(self, img, target, task_idx):
        transform, perm = self._get_transforms(task_idx)

        target = perm[target]

        img = img.view(-1, self.nx).squeeze()
        transformed = img.matmul(transform)
        output_shape = (self.seq_len,1,self.image_shape[0], self.image_shape[1])
        img = transformed.view(output_shape)

        return img, target.long()

    def _collate(self, data):
        batch_img = torch.empty((self.batch_size, self.seq_len, self.num_channels,
                                 self.image_shape[0], self.image_shape[1]),device='cpu')
        batch_targets = torch.empty((self.batch_size, self.seq_len),device='cpu')
        for i in range(0,self.batch_size*(self.seq_len), self.seq_len):
            seq = data[i:i+self.seq_len]
            img_seq, target_seq = list(zip(*seq))
            img_seq, target_seq = torch.concat(img_seq), torch.Tensor(target_seq).long()

            task_idx = np.random.randint(self.num_tasks)
            img, targets = self._apply_transforms(img_seq, target_seq, task_idx)
            
            batch_idx = i//self.seq_len 
            batch_img[batch_idx] = img
            batch_targets[batch_idx] = targets

        return batch_img, batch_targets.long()

    def _get_loader(self):
        persist = True if self.num_workers > 0 else False
        return DataLoader(self.base_dataset, 
                          batch_size=self.batch_size*(self.seq_len),
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True,collate_fn=self._collate, 
                          persistent_workers=persist, drop_last=True)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_iteration >= self.batches_per_epoch:
            self.current_iteration = 0
            self.iterator = iter(self.loader)
            raise StopIteration()
        else:
            self.current_iteration += 1
        try:
            data = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            data = next(self.iterator)
        return data
    
    def __len__(self):
        return self.batches_per_epoch

class RandomTasks:
    def __init__(self, base_dataset, num_tasks=32,
                 num_classes=10, seq_len=100, num_workers=2, device='cuda', seed_offset=0):
        assert seq_len % 2 == 0
        self.base_dataset = base_dataset
        self.num_tasks = num_tasks
        self.image_shape = base_dataset[0][0].squeeze().shape
        self.nx = self.image_shape[0] * self.image_shape[1]
        self.C = num_classes
        self.base_len = len(base_dataset)
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.base_loader = torch.utils.data.DataLoader(base_dataset, batch_size=seq_len//2, 
                                               shuffle=True, num_workers=self.num_workers, 
                                               pin_memory=True)
        self.device = device
        self.seed_offset = seed_offset

    def get_transforms(self, task_idx):
        assert task_idx < self.num_tasks
        gen = torch.Generator(device='cpu')
        gen.manual_seed(self.seed_offset + task_idx)
        a = torch.normal(0,1/self.nx, size=(self.nx,self.nx), generator=gen)
        p = torch.randperm(self.C,generator=gen)
        return a, p

    def get_seqs_from_task(self, task_idxs, batch_size, num_channels=1, seq_len=100):
        batch_img = torch.empty((batch_size, seq_len//2, num_channels, 
                                 self.image_shape[0], self.image_shape[1]),device='cpu')
        batch_targets = torch.empty((batch_size, seq_len//2),device='cpu')

        for i, (images, targets) in enumerate(self.base_loader):
            if i >= batch_size:
                break
            transform, perm = self.get_transforms(int(task_idxs[i]))
            targets = perm[targets]
            images = images.view(-1, self.nx).squeeze()
            transformed = images.matmul(transform)
            output_shape = (seq_len//2,1,self.image_shape[0], self.image_shape[1])
            img = transformed.view(output_shape)

            batch_img[i] = img
            batch_targets[i] = targets
            
        return batch_img.to(self.device), batch_targets.long().to(self.device)

    def get_batch(self, batch_size,num_channels=1):
        task_idxs = np.random.randint(self.num_tasks,size=(batch_size,))
        
        return self.get_seqs_from_task(task_idxs, batch_size, num_channels=num_channels,seq_len=self.seq_len)

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

def gen_linreg_data(seed,batch_size=64,dim=10,n_samples=50,mean=0,std=1, ws=None, device='cuda',noise_std=None):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    xs = torch.randn(batch_size, n_samples, dim, generator=gen, device=device)

    if ws is None:
        ws = mean + std*torch.randn(batch_size, dim, generator=gen, device=device)

    ys = torch.einsum('bsd,bd->bs',xs,ws).unsqueeze(-1)
    if noise_std is not None:
        ys += noise_std*torch.randn(batch_size, generator=gen)
    ys = torch.concat((ys, torch.zeros(batch_size,n_samples,dim-1,device=device)),dim=-1)

    return xs, ys, ws

# make a logistic regression task for 2-class classification
# posterior prob of class c1: p(c1|x) = sigmoid(w^T phi(x))
# phi(x) is the feature vector of input x
# prob_c1 = nn.Sigmoid(w.T @ phi)
# try without feature vectors: just sigma(w^T x)
def gen_logreg_data(seed, batch_size=64, dim=10, n_samples=50, mean=0, std=1, ws=None, device='cuda', noise_std=None):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    xs = torch.randn(batch_size, n_samples, dim, generator=gen, device=device)

    if ws is None:
        ws = mean + std*torch.randn(batch_size, dim, 1, generator=gen, device=device)

    ys = F.sigmoid(xs @ ws)

    if noise_std is not None:
        ys += noise_std*torch.randn(batch_size, generator=gen)
    ys = torch.concat((ys, torch.zeros(batch_size,n_samples,dim-1,device=device)),dim=-1)

    return xs, ys, ws

def get_transformed_seq(images, labels, num_classes, seed=None):
    '''
    Apply a random transformation to a sequence of images and randomly permute their labels. Can use a 
    specific seed or a random seed. Used in the training loop to augment the dataset. The seed is the task number.

    Args:
        images: sequence of mnist images, shape (seq_len, 1, img_size, img_size).
        labels: labels of the images, shape (seq_len).
        num_classes: number of classes in the dataset.
        seed: random seed for the transformation, default is None.

    Returns:
        transform_images: torch tensor of shape (seq_len, 1, img_size, img_size) - seq of transformed images.
        perm_labels: torch tensor of shape (seq_len) - randomly permuted labels.
        seed: seed used for the transformation.
    '''
    device = images.device
    gen = torch.Generator()
    if seed == None:
        seed = gen.initial_seed()
    else:
        gen.manual_seed(seed)

    nx = images.shape[-2] * images.shape[-1]

    transform_mtx = torch.normal(0, 1/nx, size=(nx, nx), generator=gen).to(device)

    # bug: random labels
    # perm = torch.randperm(labels.shape[0], generator=gen)
    # perm_labels = labels[perm]

    # print('labels shape', labels.shape)
    # print('labels.shape[0]', labels.shape[0])
    # print('perm', perm)

    # correct: permuted classes
    # perm = torch.randperm(num_classes, generator=gen).to(device)
    # perm_labels = perm[labels]

    # print('perm', perm)
    # print('perm_labels', perm_labels)

    # test for sequence length 1 with true labels
    perm_labels = labels

    # matmul will broadcast the transform_mtx to each image in the seq
    transform_images = torch.matmul(images.view(-1, nx), transform_mtx).view(images.shape).to(device)

    return transform_images, perm_labels, seed

def load_mnist(img_size, seq_len=100):
    '''
    Load mnist and create torch dataloader.

    Args:
        img_size: Size of the mnist image, default is 28. If img_size < 28, the image will be 
        resized via transforms.Resize (interpolation).
        seq_len: Length of each sequence in the dataloader (here, a sequence acts as a batch of images)

    Returns:
        trainloader: torch dataloader of training data
        testloader: torch dataloader of test data
    '''
    #load MNIST
    transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=seq_len, shuffle=True, 
                                              num_workers=8, pin_memory=True)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=seq_len, shuffle=False, 
                                             num_workers=8, pin_memory=True)
    return trainloader, testloader

def load_mnist_01(img_size, seq_len=100, shuffled=False):
    '''
    Load just 1s and 0s from mnist and create torch dataloader.

    Args:
        img_size: Size of the mnist image, default is 28. If img_size < 28, the image will be 
        resized via transforms.Resize (interpolation).
        seq_len: Length of each sequence in the dataloader (here, a sequence acts as a batch of images)

    Returns:
        trainloader: torch dataloader of training data
        testloader: torch dataloader of test data
    '''
    #load MNIST
    transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=transform)

    # just train on 0s and 1s from mnist
    idx = (train_data.targets == 0) | (train_data.targets == 1)
    train_data.targets = train_data.targets[idx]
    train_data.data = train_data.data[idx]

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=seq_len, shuffle=True, 
                                              num_workers=8, pin_memory=True)

    # just test on 0s and 1s from mnist
    idx = (test_data.targets == 0) | (test_data.targets == 1)
    test_data.targets = test_data.targets[idx]
    test_data.data = test_data.data[idx]

    testloader = torch.utils.data.DataLoader(test_data, batch_size=seq_len, shuffle=shuffled, 
                                             num_workers=8, pin_memory=True)

    print('shuffle=', shuffled)
    return trainloader, testloader

def make_batch(images, labels, device, n_tasks, task_list, batch_size, num_classes):
    '''
    Make a batch of batch_size num of sequences (that are length seq_len) that are transformed by get_transformed_seq.

    Args:
        images: sequence of mnist images, shape (seq_len, 1, img_size, img_size).
        labels: labels of the images, shape (seq_len).
        device: torch device ('cuda' or 'cpu')
        n_tasks: number of tasks to augment the dataset with
        task_list: list to append the seed of each task to
        batch_size: number of sequences in the batch

    Returns:
        temp_images: torch tensor of shape (batch_size, seq_len, 1, img_size, img_size) of transformed images
        temp_labels: torch tensor of shape (batch_size, seq_len) of randomly permuted labels
        task_list: list of seeds used for each task
    '''
    # add batch dimension
    images = torch.unsqueeze(images, 0).to(device)
    labels = torch.unsqueeze(labels, 0).to(device)
    # transform a batch, using a randomly selected task num from our list of n_tasks for each sequence in the batch, 
    # concatenated along the batch dimension
    temp_images, temp_labels, curr_seed = get_transformed_seq(images, labels, num_classes, seed=np.random.randint(n_tasks))
    task_list.append(curr_seed)
    for _ in range(1, batch_size):
        temp2_images, temp2_labels, curr_seed = get_transformed_seq(images, labels, num_classes, seed=np.random.randint(n_tasks))
        temp_images = torch.cat((temp_images, temp2_images), 0)
        temp_labels = torch.cat((temp_labels, temp2_labels), 0)
        task_list.append(curr_seed)

        del temp2_images, temp2_labels
    return temp_images, temp_labels, task_list

def make_unseen_batch(images, labels, device, n_tasks, test_task_list, batch_size, num_classes):
    '''
    Make a batch of batch_size num of sequences (that are length seq_len) that are transformed by get_transformed_seq using
    tasks numbers (random seeds) that were not used in training. This is done by shifting the task num by n_tasks.

    Args:
        images: sequence of mnist images, shape (seq_len, 1, img_size, img_size).
        labels: labels of the images, shape (seq_len).
        device: torch device ('cuda' or 'cpu')
        n_tasks: number of tasks to augment the dataset with
        task_list: list to append the seed of each task to
        batch_size: number of sequences in the batch

    Returns:
        temp_images: torch tensor of shape (batch_size, seq_len, 1, img_size, img_size) of transformed images
        temp_labels: torch tensor of shape (batch_size, seq_len) of randomly permuted labels
        test_task_list: list of seeds used for each task
    '''
    # add batch dimension
    images = torch.unsqueeze(images, 0).to(device)
    labels = torch.unsqueeze(labels, 0).to(device)
    # shift the task num by n_tasks to get an unseen task
    #ensure shift is by at least 2**20 to avoid overlap with training tasks
    temp_images, temp_labels, curr_seed = get_transformed_seq(images, labels, num_classes, seed=np.random.randint(n_tasks)+2**20)
    test_task_list.append(curr_seed)
    for _ in range(1, batch_size):
        temp2_images, temp2_labels, curr_seed = get_transformed_seq(images, labels, num_classes, seed=np.random.randint(n_tasks)+2**20)
        temp_images = torch.cat((temp_images, temp2_images), 0)
        temp_labels = torch.cat((temp_labels, temp2_labels), 0)
        test_task_list.append(curr_seed)

        del temp2_images, temp2_labels
    return temp_images, temp_labels, test_task_list

def load_mnist_even_odd(img_size, seq_len=100):
    '''
    Load data from mnist and separate into evens and odds and create torch dataloader.

    Args:
        img_size: Size of the mnist image, default is 28. If img_size < 28, the image will be 
        resized via transforms.Resize (interpolation).
        seq_len: Length of each sequence in the dataloader (here, a sequence acts as a batch of images)

    Returns:
        trainloader: torch dataloader of training data
        testloader: torch dataloader of test data
    '''
    #load MNIST
    transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=transform)

    # if the label is even, set it to 0, else set it to 1
    train_data.targets = (train_data.targets % 2)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=seq_len, shuffle=True, 
                                              num_workers=8, pin_memory=True)

    test_data.targets = (test_data.targets % 2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=seq_len, shuffle=False, 
                                             num_workers=8, pin_memory=True)
    return trainloader, testloader

def load_mnist_3classes(img_size, seq_len=100):
    '''
    Load just 0, 1, 2 from MNIST and create torch dataloader.

    Args:
        img_size: Size of the mnist image, default is 28. If img_size < 28, the image will be 
        resized via transforms.Resize (interpolation).
        seq_len: Length of each sequence in the dataloader (here, a sequence acts as a batch of images)

    Returns:
        trainloader: torch dataloader of training data
        testloader: torch dataloader of test data
    '''
    #load MNIST
    transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=transform)

    # only train and test on 0, 1, 2
    idx = (train_data.targets == 0) | (train_data.targets == 1) | (train_data.targets == 2)
    train_data.targets = train_data.targets[idx]
    train_data.data = train_data.data[idx]
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=seq_len, shuffle=True, 
                                              num_workers=8, pin_memory=True)

    idx = (test_data.targets == 0) | (test_data.targets == 1) | (test_data.targets == 2)
    test_data.targets = test_data.targets[idx]
    test_data.data = test_data.data[idx]
    testloader = torch.utils.data.DataLoader(test_data, batch_size=seq_len, shuffle=False, 
                                             num_workers=8, pin_memory=True)
    return trainloader, testloader

def load_mnist_5classes(img_size, seq_len=100):
    '''
    Load just 0, 1, 2, 3, 4 from MNIST and create torch dataloader.

    Args:
        img_size: Size of the mnist image, default is 28. If img_size < 28, the image will be 
        resized via transforms.Resize (interpolation).
        seq_len: Length of each sequence in the dataloader (here, a sequence acts as a batch of images)

    Returns:
        trainloader: torch dataloader of training data
        testloader: torch dataloader of test data
    '''
    #load MNIST
    transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=transform)

    # only train and test on 0, 1, 2
    idx = (train_data.targets == 0) | (train_data.targets == 1) | (train_data.targets == 2) | (train_data.targets == 3) | (train_data.targets == 4)
    train_data.targets = train_data.targets[idx]
    train_data.data = train_data.data[idx]
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=seq_len, shuffle=True, 
                                              num_workers=8, pin_memory=True)

    idx = (test_data.targets == 0) | (test_data.targets == 1) | (test_data.targets == 2) | (test_data.targets == 3) | (test_data.targets == 4)
    test_data.targets = test_data.targets[idx]
    test_data.data = test_data.data[idx]
    testloader = torch.utils.data.DataLoader(test_data, batch_size=seq_len, shuffle=False, 
                                             num_workers=8, pin_memory=True)
    return trainloader, testloader