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

#instantiate model
# block size is the seq_len * 2 to account for image label interleaving
# reduce num of classes from 10 to 2 for just 0s and 1s
model = transformer.ImageICLTransformer(d_model=64, n_layer=1, device='cuda', block_size=2*seq_len, num_classes=2)

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

losses = []
testlosses = []
testbatchlosses = []
task_list = []
test_task_list = []
accuracies = []
img_embed_grad_norm_list = []
label_embed_grad_norms_list = []
grad_norms_list = []
total_grad_norms_list = []
optimizer.zero_grad()
epochs = int(np.ceil(total_steps/len(trainloader)))

# training loop
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
        # calc norm of gradients
        img_embed_grad_norm = 0
        for p in model.img_embed.parameters():
            if p.grad is not None:
                img_embed_grad_norm += p.grad.data.norm(2).item()**2
        label_embed_grad_norm = 0
        for p in model.label_embed.parameters():
            if p.grad is not None:
                label_embed_grad_norm += p.grad.data.norm(2).item()**2
        grad_norm = 0
        total_grad_norm = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm(2).item()**2
                if 'embed' in name.lower():
                # ignore the embedding parameters
                    continue
                else:
                    grad_norm += p.grad.norm(2).item()**2
        img_embed_grad_norm_list.append(img_embed_grad_norm)
        label_embed_grad_norms_list.append(label_embed_grad_norm)
        grad_norms_list.append(grad_norm)
        total_grad_norms_list.append(total_grad_norm)
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
    # checkpoint every 10 epochs
    if epoch % 10 == 0:
        checkpoint(model, epoch, optimizer, loss.item(), run_name)
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
np.save(f'{datadir}img_embed_grad_norms_{run_name}', img_embed_grad_norm_list)
np.save(f'{datadir}label_embed_grad_norms_{run_name}', label_embed_grad_norms_list)
np.save(f'{datadir}grad_norms_{run_name}', grad_norms_list)
np.save(f'{datadir}total_grad_norms_{run_name}', total_grad_norms_list)

# make x values for number of steps per epoch to plot test loss at end of epoch on the same plot as train batch loss
# x = np.arange(len(trainloader), len(trainloader)*epochs+1, len(trainloader))
steps_per_epoch = len(trainloader)
x_initial = np.arange(1, steps_per_epoch+1)
x_remaining = np.arange(steps_per_epoch*2, steps_per_epoch+1 + (len(accuracies) - steps_per_epoch)*steps_per_epoch, steps_per_epoch)
x = np.concatenate((x_initial, x_remaining))

plot_img_grad = []
plot_label_grad = []
plot_grad = []

for i in range(len(img_embed_grad_norm_list)):
    plot_img_grad.append(img_embed_grad_norm_list[i]/total_grad_norms_list[i])
    plot_label_grad.append(label_embed_grad_norms_list[i]/total_grad_norms_list[i])
    plot_grad.append(grad_norms_list[i]/total_grad_norms_list[i])

plt.figure()
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(losses, label='train')
ax1.plot(x, testlosses, zorder=5, label='test')
ax2 = ax1.twinx()
ax2.plot(plot_img_grad, label='img embed grad norm fraction', color='red', alpha=0.5)
ax2.plot(plot_label_grad, label='label embed grad norm fraction', color='green', alpha=0.5)
ax2.plot(plot_grad, label='model grad norm (excluding embeds) fraction', color='purple', alpha=0.5)
ax2.set_yscale('log')
plt.title(f'loss - img size:({img_size}x{img_size}), tasks:2**{task_exp}')
plt.xlabel('batch')
plt.ylabel('loss')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2)
plt.tight_layout()
plt.savefig(f'{figdir}loss{run_name}.pdf')

plt.clf()
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