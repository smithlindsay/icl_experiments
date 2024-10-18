import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from einops import rearrange
import argparse
sys.path.insert(1, '/scratch/gpfs/ls1546/icl_experiments/')
import transformer
import dataset_utils
from tqdm.auto import tqdm
import pickle

parser = argparse.ArgumentParser(description='just labels')
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
n_tasks = 2**task_exp # these will be the num of seeds we send into get_transformed_batch
batch_size = args.batch_size
seq_len = args.seq_len #this is what we previously called "batch_size"
steps = args.train_steps
lr = args.lr
datadir = args.datadir
path = args.path
figdir = f'{path}figures/'
momentum = 0.9
img_size = args.img_size

trainloader, testloader = dataset_utils.load_mnist_01(img_size, seq_len=seq_len)
total_steps = int(np.ceil(steps/len(trainloader)))*len(trainloader)
device = 'cuda'

model = transformer.LabelsTransformer(d_model=64, n_layer=4, device='cuda', block_size=2*seq_len, num_classes=2)
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
avg_batch_losses = []

optimizer.zero_grad()
epochs = int(np.ceil(total_steps/len(trainloader)))
num_classes = 2

# training loop
for epoch in (pbar := tqdm(range(1, epochs + 1))):
    for images, labels in trainloader:
        model.train()
        _, temp_labels, task_list = dataset_utils.make_batch(images, labels, device, n_tasks, task_list, batch_size, num_classes)
        pred = model(temp_labels) # pred is the predictions of input tokens 1 thru T
        # train by predicting on all tokens, not just the last one
        pred = rearrange(pred, 'b t ... -> (b t) ...')
        # we want the labels for tokens 1 thru T since we can't pred token 0
        labels = rearrange(temp_labels[:, 1:], 'b t -> (b t)')
        crit = torch.nn.CrossEntropyLoss(reduction='none')
        # save losses for each batch for loss vs. num examples
        all_losses = rearrange(crit(pred, labels), "(b t) -> b t", b=batch_size).cpu().detach().numpy()
        avg_batch_losses.append(np.mean(all_losses, axis=0))
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
                test_epochs = 5
                accuracy = 0
                for test_epoch in range(1, test_epochs + 1):
                    correct = 0
                    for images, labels in testloader:
                        _, temp_labels, test_task_list = dataset_utils.make_unseen_batch(images, labels, device, n_tasks, test_task_list, batch_size, num_classes)
                        outputs = model(temp_labels)
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
    # compute the test/val loss
    if epoch != 1:
        model.eval()
        with torch.no_grad():
            testloss = 0
            num_batch = len(testloader)
            test_epochs = 5
            accuracy = 0
            for test_epoch in range(1, test_epochs + 1):
                correct = 0
                for images, labels in testloader:
                    _, temp_labels, test_task_list = dataset_utils.make_unseen_batch(images, labels, device, n_tasks, test_task_list, batch_size, num_classes)
                    outputs = model(temp_labels)
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
# np.save(f'{datadir}avg_batch_losses_{run_name}', avg_batch_losses)
# save avg_batch_losses as pickle
with open(f'{datadir}avg_batch_losses_{run_name}.pkl', 'wb') as f:
    pickle.dump(avg_batch_losses, f)
# make x values for number of steps per epoch to plot test loss at end of epoch on the same plot as train batch loss
# x = np.arange(len(trainloader), len(trainloader)*epochs+1, len(trainloader))
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