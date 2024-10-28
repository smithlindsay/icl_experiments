import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import sys
from einops import rearrange
import argparse
sys.path.insert(1, '/scratch/gpfs/ls1546/icl_experiments/')
import dataset_utils
from tqdm.auto import tqdm

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BlockGroup(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=2, stride=1):
        super(BlockGroup, self).__init__()
        first = BasicBlock(in_channels,out_channels,stride=stride)
        blocks = [first] + [BasicBlock(out_channels,out_channels,stride=stride) for i in range(n_blocks-1)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self,x):
        out = x
        return self.blocks(out)

# add mlp layer to end of resnet to reduce dimensions to 1
class Resnet(nn.Module):
    def __init__(self, in_channels, conv1_channels=8, channels_per_group=[16,32,32,64], blocks_per_group=[2,2,2,2], strides=[1,1,1,1]):
        super().__init__()
        assert len(channels_per_group) == len(blocks_per_group), "list dimensions must match"
        assert len(blocks_per_group) == len(strides), "list dimensions must match"
        assert len(channels_per_group) >= 1, "must have at least one layer"

        self.conv1 = nn.Conv2d(in_channels, conv1_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        layers = []
        for i, (c,b,s) in enumerate(zip(channels_per_group, blocks_per_group, strides)):
            if i == 0:
                layers.append(BlockGroup(conv1_channels,c,n_blocks=b,stride=s))
            else:
                prev_c = channels_per_group[i-1]
                layers.append(BlockGroup(prev_c,c,n_blocks=b,stride=s))
        self.layers = nn.Sequential(*layers)
        self.final = nn.Linear(channels_per_group[-1], 2)

    def forward(self,x):
        out = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        #average over superpixels to create embedding
        out = torch.mean(out,axis=(2,3))
        out = self.final(F.relu(out))
        return out
    
task_exp = 16
n_tasks = 2**task_exp # these will be the num of seeds we send into get_transformed_batch
batch_size = 32
seq_len = 1 #this is what we previously called "batch_size"
steps = 15000
lr = 3e-4
datadir = '/scratch/gpfs/ls1546/icl_experiments/data/'
path = '/scratch/gpfs/ls1546/icl_experiments/'
figdir = f'{path}figures/'
momentum = 0.9
img_size = 16

d_model = 64    
c_per_g = [16,32,32,d_model]
resnet = Resnet(1, channels_per_group=c_per_g)
# will output (batch_size, d_model)

trainloader, testloader = dataset_utils.load_mnist_01(img_size, seq_len=seq_len)
total_steps = int(np.ceil(steps/len(trainloader)))*len(trainloader)
device = 'cuda'

model = resnet.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])

losses = []
testlosses = []
testbatchlosses = []
task_list = []
test_task_list = []
accuracies = []
optimizer.zero_grad()
epochs = int(np.ceil(total_steps/len(trainloader)))
num_classes = 2

def get_transformed_seq(image, seed=None):
    device = images.device
    gen = torch.Generator()
    if seed == None:
        seed = gen.initial_seed()
    else:
        gen.manual_seed(seed)

    nx = image.shape[-2] * image.shape[-1]

    transform_mtx = torch.normal(0, 1/nx, size=(nx, nx), generator=gen).to(device)

    # matmul will broadcast the transform_mtx to each image in the seq
    transform_image = torch.matmul(image.view(-1, nx), transform_mtx).view(images.shape).to(device)

    return transform_image, seed


for epoch in (pbar := tqdm(range(1, epochs + 1))):
    for images, labels in trainloader:
        model.train()
        # grab a batch of images and labels
        # trasform each image in the batch individually as a separate task
        for i in range(images.shape[0]):
            images[i], seed = get_transformed_seq(images[i], seed=np.random.randint(n_tasks))
            task_list.append(seed)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        pbar.set_description(f"epoch: {epoch}, loss: {loss.item():4f}")

        # temp_images, temp_labels, task_list = dataset_utils.make_batch(images, labels, device, n_tasks, task_list, batch_size, num_classes)
        # temp_images = temp_images.squeeze(1)
        # outputs = model(temp_images)
        # pred = outputs
        # # train by predicting on all tokens, not just the last one
        # labels = rearrange(temp_labels, 'b t -> (b t)')
        # loss = criterion(pred, labels)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad() 
        # losses.append(loss.item())
        # pbar.set_description(f"epoch: {epoch}, loss: {loss.item():4f}")

        # # if first epoch, calc the test loss and acc at each step
        # if epoch == 1:
        #     model.eval()
        #     with torch.no_grad():
        #         testloss = 0
        #         num_batch = len(testloader)
        #         test_epochs = 1
        #         accuracy = 0
        #         for test_epoch in range(1, test_epochs + 1):
        #             correct = 0
        #             for images, labels in testloader:
        #                 temp_images, temp_labels, test_task_list = dataset_utils.make_unseen_batch(images, labels, device, n_tasks, test_task_list, batch_size, num_classes)
        #                 temp_images = temp_images.squeeze(1)
        #                 outputs = model(temp_images)
        #                 pred = outputs
        #                 temp_labels = rearrange(temp_labels, 'b t -> (b t)')
        #                 batchloss = criterion(pred,temp_labels)
        #                 _, predicted = torch.max(pred, 1)
        #                 correct += (predicted == temp_labels).sum()
        #                 testloss += batchloss.item()
        #                 testbatchlosses.append(batchloss.item())
        #             accuracy += 100 * (correct.item()) / (batch_size*num_batch)
        #         accuracy /= test_epochs
        #         accuracies.append(accuracy)
        #         testloss /= (len(testloader) * test_epochs)
        #         testlosses.append(testloss)
    # test the model
    model.eval()
    with torch.no_grad():
        testloss = 0
        num_batch = len(testloader)
        accuracy = 0
        correct = 0
        for images, labels in testloader:
            temp_images, temp_labels, test_task_list = dataset_utils.make_unseen_batch(images, labels, device, n_tasks, test_task_list, batch_size, num_classes)
            temp_images = temp_images.squeeze(1)
            outputs = model(temp_images)
            pred = outputs
            temp_labels = rearrange(temp_labels, 'b t -> (b t)')
            batchloss = criterion(pred,temp_labels)
            _, predicted = torch.max(pred, 1)
            correct += (predicted == temp_labels).sum()
            testloss += batchloss.item()
            testbatchlosses.append(batchloss.item())
        accuracy += 100 * (correct.item()) / (batch_size*num_batch)
        accuracies.append(accuracy)
        testloss /= len(testloader)
        testlosses.append(testloss)

print(correct.item())
print(accuracy)
run_name = 'resnetseqlen1is16t16'
torch.save(model, f"{datadir}{run_name}.pt")

losses = np.array(losses)
np.save(f'{datadir}losses_{run_name}', losses)
testlosses = np.array(testlosses)
np.save(f'{datadir}testlosses_{run_name}', testlosses)
accuracies = np.array(accuracies)
np.save(f'{datadir}accuracies_{run_name}', accuracies)