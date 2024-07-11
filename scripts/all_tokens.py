import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
from einops import rearrange
import argparse
sys.path.insert(1, '/scratch/gpfs/ls1546/icl_experiments/')
import transformer
import dataset_utils
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='Train a transformer model on 2 classes of MNIST for ICL')
parser.add_argument('--task_exp', type=int, default=16, help='exp of 2**___: number of tasks to augment the dataset with')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--seq_len', type=int, default=100, help='sequence length')
parser.add_argument('--train_steps', type=int, default=30_000, help='min number of training steps')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--datadir', type=str, default='/scratch/gpfs/ls1546/icl_experiments/data/')
parser.add_argument('--path', type=str, default='/scratch/gpfs/ls1546/icl_experiments/', help='path to icl dir')
parser.add_argument('--img_size', type=int, default=28, help='size of the MNIST image')
args = parser.parse_args()


task_exp = args.task_exp
run_name = f'{args.run_name}is{args.img_size}t{task_exp}'
# these will be the num of seeds we send into get_transformed_batch
n_tasks = 2**task_exp
batch_size = args.batch_size
#this is what we previously called "batch_size"
seq_len = args.seq_len
steps = args.train_steps
lr = args.lr
datadir = args.datadir
path = args.path
figdir = f'{path}figures/'
momentum = 0.9
img_size = args.img_size

#load MNIST
transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('../data', train=False, transform=transform)

# just train on 0s and 1s from mnist
idx = (train_data.targets == 0) | (train_data.targets == 1)
train_data.targets = train_data.targets[idx]
train_data.data = train_data.data[idx]

trainloader = torch.utils.data.DataLoader(train_data, batch_size=seq_len, shuffle=True)

# just test on 0s and 1s from mnist
idx = (test_data.targets == 0) | (test_data.targets == 1)
test_data.targets = test_data.targets[idx]
test_data.data = test_data.data[idx]

testloader = torch.utils.data.DataLoader(test_data, batch_size=seq_len, shuffle=False)

total_steps = int(np.ceil(steps/len(trainloader)))*len(trainloader)
device = 'cuda'

#instantiate model
# block size is the seq_len * 2 to account for image label interleaving
# reduce num of classes from 10 to 2 for just 0s and 1s
model = transformer.ImageICLTransformer(d_model=64, n_layer=4, device='cuda', block_size=2*seq_len, num_classes=2)

#train the model to do MNIST classification

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])

def checkpoint(model, epoch, optimizer, loss, run_name):
    path = f"{datadir}ckpt{run_name}{epoch}.pt"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

def make_seq(images, labels, device, n_tasks, task_list, batch_size):
    # add batch dimension
    images = torch.unsqueeze(images, 0).to(device)
    labels = torch.unsqueeze(labels, 0).to(device)
    # transform a batch, using a randomly selected task num from our list of n_tasks for each sequence in the batch, 
    # concatenated along the batch dimension
    temp_images, temp_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, seed=np.random.randint(n_tasks))
    task_list.append(curr_seed)
    for _ in range(1, batch_size):
        temp2_images, temp2_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, seed=np.random.randint(n_tasks))
        temp_images = torch.cat((temp_images, temp2_images), 0)
        temp_labels = torch.cat((temp_labels, temp2_labels), 0)
        task_list.append(curr_seed)

    # images.shape = batch, seq, 1, 28, 28
    # labels.shape = batch, seq
    del temp2_images, temp2_labels
    return temp_images, temp_labels

def make_unseen_seq(images, labels, device, n_tasks, test_task_list, batch_size):
    # add batch dimension
    images = torch.unsqueeze(images, 0).to(device)
    labels = torch.unsqueeze(labels, 0).to(device)
    # shift the task num by n_tasks to get an unseen task
    temp_images, temp_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, 
                                                                              seed=np.random.randint(n_tasks)+n_tasks)
    test_task_list.append(curr_seed)
    for _ in range(1, batch_size):
        temp2_images, temp2_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, 
                                                                                    seed=np.random.randint(n_tasks)+n_tasks)
        temp_images = torch.cat((temp_images, temp2_images), 0)
        temp_labels = torch.cat((temp_labels, temp2_labels), 0)
        test_task_list.append(curr_seed)

    del temp2_images, temp2_labels
    return temp_images, temp_labels

losses = []
testlosses = []
testbatchlosses = []
task_list = []
test_task_list = []
optimizer.zero_grad()
epochs = int(np.ceil(total_steps/len(trainloader)))
# training loop
for epoch in (pbar := tqdm(range(1, epochs + 1))):
    model.train()
    for images, labels in trainloader:
        temp_images, temp_labels = make_seq(images, labels, device, n_tasks, task_list, batch_size)
        outputs = model((temp_images,temp_labels))
        pred = outputs[:, 0::2, :]
        del outputs
        # train by predicting on all tokens, not just the last one
        pred = rearrange(pred, 'b h ... -> (b h) ...')
        labels = rearrange(temp_labels, 'b c -> (b c)')
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        losses.append(loss.item())
        pbar.set_description(f"epoch: {epoch}, loss: {loss.item():4f}")
    # checkpoint every 10 epochs
    if epoch % 10 == 0:
        checkpoint(model, epoch, optimizer, loss.item(), run_name)
    # compute the test/val loss
    model.eval()
    with torch.no_grad():
        correct = 0
        testloss = 0
        num_batch = len(testloader)
        for images, labels in testloader:
            temp_images, temp_labels = make_unseen_seq(images, labels, device, n_tasks, test_task_list, batch_size)
            outputs = model((temp_images,temp_labels))
            pred = outputs[:,-1,:]
            batchloss = criterion(pred,temp_labels[:,-1])
            _, predicted = torch.max(pred, 1)
            correct += (predicted == temp_labels[:,-1]).sum()
            testloss += batchloss.item()
            testbatchlosses.append(batchloss.item())
        accuracy = 100 * (correct.item()) / (batch_size*num_batch)
        testloss /= len(testloader)
        testlosses.append(testloss)

print(correct.item())
print(accuracy)

torch.save(model, f"{datadir}{run_name}.pt")

losses = np.array(losses)
np.save(f'{datadir}losses_{run_name}', losses)
testlosses = np.array(testlosses)
np.save(f'{datadir}testlosses_{run_name}', testlosses)

# make x values for number of steps per epoch to plot test loss at end of epoch on the same plot as train batch loss
x = np.arange(len(trainloader), len(trainloader)*epochs+1, len(trainloader))

plt.figure()
plt.plot(losses, label='train')
plt.plot(x, testlosses, zorder=5, label='test')
plt.title(f'loss - img size:({img_size}x{img_size}), tasks:2**{task_exp}')
plt.xlabel('batch')
plt.ylabel('loss')
plt.legend()
plt.savefig(f'{figdir}loss{run_name}.pdf')