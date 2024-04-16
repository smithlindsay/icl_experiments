import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import embedding #these lines get underlined with an error but they work
import transformer
import dataset_utils
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser(description="In-context learning on randomized MNIST")
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='learning rate (default: 3e-4)')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum (default: 0.9)')
parser.add_argument('-s', '--steps', default=50000, type=int,
                    metavar='N', help='total steps to take (default: 50,000)')
parser.add_argument('--device', default='cuda', type=str, help='device to run the training script on')
parser.add_argument('--seq_len', '--sl', default=200, type=int, 
                    help='number of tokens to send into the transformer (default: 200)')
parser.add_argument('--n-tasks', '--num-tasks', default=16, type=int)
parser.add_argument('--test-batches', default=50, type=int)

args = parser.parse_args()


#load MNIST
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
train_data = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
test_data = datasets.MNIST('../data', train=False,
                    transform=transform)

batch_size = args.batch_size
lr = args.lr
momentum = args.momentum
steps = args.steps
device = args.device
seq_len = args.seq_len

#augment MNIST with multiple tasks
n_tasks = args.n_tasks
task_gen = dataset_utils.RandomTasks(train_data, num_tasks=n_tasks, seq_len=seq_len)

model = transformer.ImageICLTransformer(d_model=64,device=device,block_size=seq_len,n_layer=4)
print(model)

#train the model to do MNIST classification

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])

loss_history = [0]*steps
model.train()
data_avg_t = 0
loss_avg_t = 0
for step in tqdm(range(steps)):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    images, labels = task_gen.get_batch(batch_size)
    end.record()
    torch.cuda.synchronize()
    data_avg_t += start.elapsed_time(end)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    optimizer.zero_grad()
    outputs = model((images,labels))
    pred = outputs[:,-1,:]
    loss = criterion(pred,labels[:,-1])
    loss.backward()
    optimizer.step()
    end.record()
    torch.cuda.synchronize()
    loss_avg_t += start.elapsed_time(end)
    loss_history[step] = loss.item()

loss_history = np.array(loss_history)
plt.plot(loss_history)
plt.savefig('loss_history.png')
print("\nDATA: ", data_avg_t/steps)
print("LOSS: ", loss_avg_t/steps)

def test_model(model, task_generator, batch_size=32, seq_len=50, num_batch=50, device='cuda'):
    correct = 0
    model.eval()
    for batch in tqdm(range(num_batch)):
        images, labels = task_generator.get_batch(batch_size)
        outputs = model((images,labels))
        pred = outputs[:,-1,:]
        loss = criterion(pred,labels[:,-1])
        _, predicted = torch.max(pred, 1)
        correct += (predicted == labels[:,-1]).sum()
    accuracy = 100 * (correct.item()) / (batch_size*num_batch)
    print(accuracy)

test_task_gen = dataset_utils.RandomTasks(test_data, num_tasks=n_tasks, seq_len=seq_len, seed_offset=n_tasks)
test_model(model, test_task_gen, batch_size=batch_size, num_batch=args.test_batches, seq_len=seq_len)

torch.save(model.state_dict(), "final.th")




