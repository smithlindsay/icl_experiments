import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import embedding
import transformer
import dataset_utils
from tqdm.auto import tqdm
import sys
import argparse
sys.path.insert(1, '/scratch/gpfs/ls1546/icl_experiments/')

parser = argparse.ArgumentParser(description='Train a transformer model on MNIST for ICL')
parser.add_argument('--n_tasks_exp', type=int, default=14, help='exp of 2**___: number of tasks to augment the dataset with')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--seq_len', type=int, default=100, help='sequence length')
parser.add_argument('--train_steps', type=int, default=40_000, help='min number of training steps')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--run_name', type=str, default='')
args = parser.parse_args()

run_name = args.run_name + f'tasks{args.n_tasks_exp}bs{args.batch_size}seq{args.seq_len}'

#load MNIST
transform=transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('../data', train=False, transform=transform)

#augment MNIST with multiple tasks
# these will be the num of seeds we send into get_transformed_batch
n_tasks = 2**args.n_tasks_exp

batch_size = args.batch_size
seq_len = args.seq_len #this is what we previously called "batch_size"
steps = args.train_steps
lr = args.lr

momentum = 0.9

# just train on 0s and 1s from mnist
idx = (train_data.targets == 0) | (train_data.targets == 1)
train_data.targets = train_data.targets[idx]
train_data.data = train_data.data[idx]

trainloader = torch.utils.data.DataLoader(train_data, batch_size=seq_len, shuffle=True)

total_steps = int(np.ceil(steps/len(trainloader)))*len(trainloader)

# grad accumulation to fit in memory
if batch_size > 128 or seq_len > 100:
    acc_steps = batch_size // 128
    total_steps *= acc_steps
    eff_batch_size = batch_size
    batch_size = batch_size // acc_steps
else:
    acc_steps = 1
    eff_batch_size = None

log_freq = 100
device = 'cuda'

#instantiate model
# block size is the seq_len * 2 to account for image label interleaving
model = transformer.ImageICLTransformer(d_model=64, n_layer=4, device='cuda', block_size=2*seq_len)
print(model)

#train the model to do MNIST classification

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])

loss_history = [0]*total_steps
model.train()
data_avg_t = 0
loss_avg_t = 0

# here our "batch_size" is actually the sequence length since we want to draw seq_length samples from the dataset, and then transform them batch_num times, where each transformation is a different task the batch should have

# shape: (batch_size,seq_len,1,28,28),
# right now it has shape: (seq_len,1,28,28), and we will add in the batch dimension in the loop below

def checkpoint(model, epoch, optimizer, loss, run_name):
    path = f"ckpt{run_name}{epoch}.pt"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

step = 0
task_list = []
optimizer.zero_grad()
epochs = int(np.ceil(total_steps/len(trainloader)))
for epoch in (pbar := tqdm(range(1, epochs + 1))):
    for images, labels in tqdm(trainloader):
        # add batch dimension
        images = torch.unsqueeze(images, 0).to(device)
        labels = torch.unsqueeze(labels, 0).to(device)
        # transform a batch, using a randomly selected task num from our list of n_tasks for each sequence in the batch, 
        # concatenated along the batch dimension
        temp_images, temp_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, seed=np.random.randint(n_tasks))
        task_list.append(curr_seed)
        for i in range(1, batch_size):
            temp2_images, temp2_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, seed=np.random.randint(n_tasks))
            temp_images = torch.cat((temp_images, temp2_images), 0)
            temp_labels = torch.cat((temp_labels, temp2_labels), 0)
            task_list.append(curr_seed)

        images = temp_images
        labels = temp_labels
        del temp_images, temp_labels, temp2_images, temp2_labels

        
        outputs = model((images,labels))

        pred = outputs[:,-1,:]
        loss = criterion(pred,labels[:,-1]) / acc_steps
        loss.backward()
        if step % acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad() 

        loss_history[step] = loss.item()
        step += 1
        pbar.set_description(f"epoch: {epoch}, loss: {loss.item():4f}")
    checkpoint(model, epoch, optimizer, loss.item(), run_name)

loss_history = np.array(loss_history)[:step]
plt.figure()
plt.plot(loss_history)
if eff_batch_size:
    plt.title(f'batch loss, eff. batch size: {eff_batch_size}, seq len: {seq_len}')
else:
    plt.title(f'batch loss, batch size: {batch_size}, seq len: {seq_len}')
plt.savefig(f"loss_history{run_name}.pdf")
np.save(f"loss_history{run_name}.npy", loss_history)

del outputs, pred, images, labels
torch.save(model, f"icl_model{run_name}.pth")

def test_model(model, test_data, n_tasks, batch_size, seq_len, device='cuda'):
    correct = 0
    model.eval()
    
    test_task_list = []
    testloader = torch.utils.data.DataLoader(test_data, batch_size=seq_len, shuffle=False)

    # test on just 0s and 1s
    test_data.targets = test_data.targets[(test_data.targets == 0) | (test_data.targets == 1)]
    test_data.data = test_data.data[(test_data.targets == 0) | (test_data.targets == 1)]

    num_batch = len(testloader)

    for images, labels in tqdm(testloader):
        # add batch dimension
        images = torch.unsqueeze(images, 0).to(device)
        labels = torch.unsqueeze(labels, 0).to(device)
        temp_images, temp_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, seed=np.random.randint(n_tasks))
        test_task_list.append(curr_seed)
        for batch in range(1, batch_size):
            temp2_images, temp2_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, seed=np.random.randint(n_tasks))
            temp_images = torch.cat((temp_images, temp2_images), 0)
            temp_labels = torch.cat((temp_labels, temp2_labels), 0)
            test_task_list.append(curr_seed)

        images = temp_images
        labels = temp_labels
        del temp_images, temp_labels, temp2_images, temp2_labels

        outputs = model((images,labels))
        pred = outputs[:,-1,:]
        loss = criterion(pred,labels[:,-1])
        _, predicted = torch.max(pred, 1)
        correct += (predicted == labels[:,-1]).sum()
    accuracy = 100 * (correct.item()) / (batch_size*num_batch)
    print(correct.item())
    print(accuracy)

test_model(model, test_data, n_tasks, batch_size, seq_len, device='cuda')
