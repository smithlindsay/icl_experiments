import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import sys
sys.path.insert(1, '/scratch/gpfs/ls1546/icl_experiments/')
import dataset_utils
from tqdm.auto import tqdm
import plot_figs
from einops import rearrange

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
batch_size = 1000
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

trainloader, testloader = dataset_utils.load_mnist_01(img_size, seq_len=batch_size)
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

def make_transform_mtxs(nx, device, batch_size, n_tasks):
    transform_mtxs = torch.empty((batch_size, nx, nx), device=device)
    gen = torch.Generator()
    for i in range(batch_size):
        seed = np.random.randint(n_tasks)
        gen.manual_seed(seed)

        transform_mtxs[i, :, :] = torch.normal(0, 1/nx, size=(nx, nx), generator=gen)

    transform_mtxs.to(device)

    return transform_mtxs

for epoch in (pbar := tqdm(range(1, epochs + 1))):
    for images, labels in trainloader:
        # skip last batch if it is not full
        if images.shape[0] != batch_size:
            continue
        model.train()
        # grab a batch of images and labels
        images, labels = images.to(device), labels.to(device)
        nx = images.shape[-2] * images.shape[-1]
        transform_mtxs = make_transform_mtxs(nx, device, batch_size, n_tasks)

        # Reshape images from (batch_size, 1, 16, 16) to (batch_size, 256)
        flattened = rearrange(images, 'b c h w -> b (c h w)')

        # Use einsum for the multiplication
        # Pattern: 'b n, b n m -> b m'
        # where b=batch, n=input dims (256), m=output dims (256)
        transform_images = torch.einsum('bn,bnm->bm', flattened, transform_mtxs).view(images.shape).to(device)

        outputs = model(transform_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        pbar.set_description(f"epoch: {epoch}, loss: {loss.item():4f}")
    # test the model
    model.eval()
    with torch.no_grad():
        testloss = 0
        num_batch = len(testloader)
        accuracy = 0
        correct = 0
        for images, labels in testloader:
            # skip last batch if it is not full
            if images.shape[0] != batch_size:
                continue
            images, labels = images.to(device), labels.to(device)
            nx = images.shape[-2] * images.shape[-1]
            transform_mtxs = make_transform_mtxs(nx, device, batch_size, n_tasks)
            
            # Reshape images from (batch_size, 1, 16, 16) to (batch_size, 256)
            flattened = rearrange(images, 'b c h w -> b (c h w)')
            # Use einsum for the multiplication
            # Pattern: 'b n, b n m -> b m'
            # where b=batch, n=input dims (256), m=output dims (256)
            transform_images = torch.einsum('bn,bnm->bm', flattened, transform_mtxs).view(images.shape).to(device)

            outputs = model(transform_images)
            batchloss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
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

plot_figs.plot_loss_acc(trainloader, losses, testlosses, accuracies, figdir, run_name)