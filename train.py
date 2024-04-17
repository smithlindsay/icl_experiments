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
import torch.backends.cudnn as cudnn
import argparse
import time

parser = argparse.ArgumentParser(description="In-context learning on randomized MNIST")
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='learning rate (default: 3e-4)')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum (default: 0.9)')
parser.add_argument('-e', '--epochs', default=50, type=int,
                    metavar='N', help='total number of epochs (default: 50)')
parser.add_argument('--batches-per-epoch', default=1000, type=int, metavar='N',
                    help='number of batches/steps per epoch (test performance calculated after each epoch)')
parser.add_argument('--device', default='cuda', type=str, help='device to run the training script on')
parser.add_argument('--seq_len', '--sl', default=100, type=int, 
                    help='number of examples per sequence to send into the transformer (default: 100)')
parser.add_argument('--n-tasks', '--num-tasks', default=16, type=int)
parser.add_argument('--test-batches', default=50, type=int)
parser.add_argument('--num-workers', default=8, type=int)
parser.add_argument('--d-model', default=64, type=int)
parser.add_argument('--n-layer', default=4, type=int)

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
epochs = args.epochs
batches_per_epoch = args.batches_per_epoch
device = args.device
seq_len = args.seq_len
num_workers = args.num_workers
d_model = args.d_model
n_layer = args.n_layer

print(num_workers, " workers")

print("Running setup...")
t1 = time.time()
#augment MNIST with multiple tasks
n_tasks = args.n_tasks
seq_loader = dataset_utils.SequenceLoader(train_data, num_tasks=n_tasks, 
                                          seq_len=seq_len, batch_size=batch_size, 
                                          batches_per_epoch=batches_per_epoch,
                                          num_workers=num_workers,
                                          dataset_expansion_factor=2)
loader = seq_loader._get_loader()
t2 = time.time()
print("Setup complete, time:", (t2-t1)/60, "minutes")

model = transformer.ImageICLTransformer(d_model=d_model,device=device,
                                        block_size=seq_len*2,n_layer=n_layer)
print(model)

#train the model to do MNIST classification
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
#lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])

def test_model(model, test_loader, device='cuda', epochs=1, test_batch_size=64):
    correct = 0
    model.eval()
    for epoch in range(epochs):
        for i, (images,labels) in enumerate((test_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model((images,labels))
            pred = outputs[:,-1,:]
            loss = criterion(pred,labels[:,-1])
            _, predicted = torch.max(pred, 1)
            correct += (predicted == labels[:,-1]).sum()
    accuracy = 100 * correct.item() / (epochs*len(test_loader)*test_batch_size)
    print("\nTest Accuracy:", accuracy, "%")

loss_history = [0]*len(loader)*epochs
for epoch in range(epochs):
    print("Epoch: ", epoch)
    model.train()
    print(len(loader))
    t1 = time.time()
    for step, (images, labels) in enumerate((loader)):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model((images,labels))
        pred = outputs[:,-1,:]
        loss = criterion(pred,labels[:,-1])
        loss.backward()
        optimizer.step()
        loss_history[step+len(loader)*epoch] = loss.item()

    #print("Test:")

    #test_loader = dataset_utils.SequenceLoader(test_data, num_tasks=n_tasks, batch_size=64,
                                               #seq_len=seq_len, batches_per_epoch=300, 
                                               #num_workers=num_workers,dataset_expansion_factor=10, 
                                               #seed_offset=n_tasks)
    #test_model(model, test_loader, device=device)

    t2 = time.time()
    print("Epoch complete, time:", (t2-t1)/60, "minutes")

loss_history = np.array(loss_history)
plt.plot(loss_history)
plt.savefig('loss_history.png')

torch.save(model.state_dict(), "final.th")




