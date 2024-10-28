import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
from einops import rearrange
import argparse
sys.path.insert(1, '/scratch/gpfs/ls1546/icl_experiments/')
import embedding
import transformer
import dataset_utils
from tqdm.auto import tqdm   

task_exp = 16
n_tasks = 2**task_exp # these will be the num of seeds we send into get_transformed_batch
batch_size = 1000
seq_len = 1 #this is what we previously called "batch_size"
steps = 30000
lr = 3e-4
datadir = '/scratch/gpfs/ls1546/icl_experiments/data/'
path = '/scratch/gpfs/ls1546/icl_experiments/'
figdir = f'{path}figures/'
momentum = 0.9
img_size = 16

trainloader, testloader = dataset_utils.load_mnist_01(img_size, seq_len=seq_len)
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

losses = []
testlosses = []
testbatchlosses = []
task_list = []
test_task_list = []
accuracies = []
optimizer.zero_grad()
epochs = int(np.ceil(total_steps/len(trainloader)))
num_classes = 2

for epoch in (pbar := tqdm(range(1, epochs + 1))):
    for images, labels in trainloader:
        model.train()
        temp_images, temp_labels, task_list = dataset_utils.make_batch(images, labels, device, n_tasks, task_list, batch_size, num_classes)
        outputs = model((temp_images,temp_labels))
        pred = outputs[:, 0::2, :]
        # train by predicting on all tokens, not just the last one
        pred = rearrange(pred, 'b t ... -> (b t) ...')
        labels = rearrange(temp_labels, 'b t -> (b t)')
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
        losses.append(loss.item())
        pbar.set_description(f"epoch: {epoch}, loss: {loss.item():4f}")

        # if first epoch, calc the test loss and acc at each step
        if epoch == 1:
            model.eval()
            with torch.no_grad():
                testloss = 0
                num_batch = len(testloader)
                test_epochs = 1
                accuracy = 0
                for test_epoch in range(1, test_epochs + 1):
                    correct = 0
                    for images, labels in testloader:
                        temp_images, temp_labels, test_task_list = dataset_utils.make_unseen_batch(images, labels, device, n_tasks, test_task_list, batch_size, num_classes)
                        outputs = model((temp_images,temp_labels))
                        pred = outputs[:,-1,:]
                        batchloss = criterion(pred,temp_labels[:,-1])
                        _, predicted = torch.max(pred, 1)
                        correct += (predicted == temp_labels[:,-1]).sum()
                        testloss += batchloss.item()
                        testbatchlosses.append(batchloss.item())
                    accuracy += 100 * (correct.item()) / (batch_size*num_batch)
                accuracy /= test_epochs
                accuracies.append(accuracy)
                testloss /= (len(testloader) * test_epochs)
                testlosses.append(testloss)
    # test the model
    if epoch != 1:
        model.eval()
        with torch.no_grad():
            testloss = 0
            num_batch = len(testloader)
            test_epochs = 1
            accuracy = 0
            for test_epoch in range(1, test_epochs + 1):
                correct = 0
                for images, labels in testloader:
                    temp_images, temp_labels, test_task_list = dataset_utils.make_unseen_batch(images, labels, device, n_tasks, test_task_list, batch_size, num_classes)
                    outputs = model((temp_images,temp_labels))
                    pred = outputs[:,-1,:]
                    batchloss = criterion(pred,temp_labels[:,-1])
                    _, predicted = torch.max(pred, 1)
                    correct += (predicted == temp_labels[:,-1]).sum()
                    testloss += batchloss.item()
                    testbatchlosses.append(batchloss.item())
                accuracy += 100 * (correct.item()) / (batch_size*num_batch)
            accuracy /= test_epochs
            accuracies.append(accuracy)
            testloss /= (len(testloader) * test_epochs)
            testlosses.append(testloss)

print(correct.item())
print(accuracy)
run_name = 'seqlen1is16t16'
torch.save(model, f"{datadir}{run_name}.pt")

losses = np.array(losses)
np.save(f'{datadir}losses_{run_name}', losses)
testlosses = np.array(testlosses)
np.save(f'{datadir}testlosses_{run_name}', testlosses)
accuracies = np.array(accuracies)
np.save(f'{datadir}accuracies_{run_name}', accuracies)