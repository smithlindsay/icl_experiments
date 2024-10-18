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

task_exp = 16
n_tasks = 2**task_exp # these will be the num of seeds we send into get_transformed_batch
batch_size = 32
seq_len = 1 #this is what we previously called "batch_size"
steps = 1500
lr = 3e-4
path = '/scratch/gpfs/ls1546/icl_experiments/'
datadir = f'{path}data/'
figdir = f'{path}figures/'
momentum = 0.9
img_size = 28
run_name = f'nocontextis{img_size}t{task_exp}'

trainloader, testloader = dataset_utils.load_mnist_01(img_size, seq_len=seq_len)
total_steps = int(np.ceil(steps/len(trainloader)))*len(trainloader)
device = 'cuda'

#instantiate model
# block size is the seq_len * 2 to account for image label interleaving
# reduce num of classes from 10 to 2 for just 0s and 1s
model = transformer.ImageICLTransformer(d_model=64, n_layer=4, device='cuda', block_size=2, num_classes=2)

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

def checkpoint(model, epoch, optimizer, loss, losses, testlosses, accuracies, run_name):
    path = f"{datadir}ckpt{run_name}{epoch}.pt"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'losses': losses,
            'testlosses': testlosses,
            'accuracies': accuracies
            }, path)

# training loop
step = 0
for epoch in (pbar := tqdm(range(1, epochs + 1))):
    for images, labels in trainloader:
        model.train()
        temp_images, temp_labels, task_list = dataset_utils.make_batch(images, labels, device, n_tasks, task_list, batch_size)
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
        step += 1
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
                        temp_images, temp_labels, test_task_list = dataset_utils.make_unseen_batch(images, labels, device, n_tasks, test_task_list, batch_size)
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
        if step % 100 == 0:
            checkpoint(model, step, optimizer, loss.item(), losses, testlosses, accuracies, run_name)
    # compute the test/val loss
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
                    temp_images, temp_labels, test_task_list = dataset_utils.make_unseen_batch(images, labels, device, n_tasks, test_task_list, batch_size)
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


torch.save(model, f"{datadir}{run_name}.pt")

losses = np.array(losses)
np.save(f'{datadir}losses_{run_name}', losses)
testlosses = np.array(testlosses)
np.save(f'{datadir}testlosses_{run_name}', testlosses)
accuracies = np.array(accuracies)
np.save(f'{datadir}accuracies_{run_name}', accuracies)

steps_per_epoch = len(trainloader)
x_initial = np.arange(1, steps_per_epoch+1)
x_remaining = np.arange(steps_per_epoch*2, steps_per_epoch+1 + (len(accuracies) - steps_per_epoch)*steps_per_epoch, steps_per_epoch)
x = np.concatenate((x_initial, x_remaining))

plt.figure()
fig, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(losses, label='train')
ax3.plot(x, testlosses, zorder=5, label='test')
ax1_twin = ax3.twinx()
color = 'tab:red'
ax1_twin.plot(x, accuracies, label='test acc', color=color)
ax1_twin.tick_params(axis='y', labelcolor=color)

ax3.set_xlabel('batch')
ax3.set_ylabel('loss')
ax1_twin.set_ylabel('test accuracy', color=color)
# set twin y axis scale to 0 - 100
ax1_twin.set_ylim(0, 100)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()


ax3.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2)
plt.tight_layout()
plt.savefig(f'{figdir}loss{run_name}_acc.pdf')