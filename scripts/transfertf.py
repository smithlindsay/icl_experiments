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
import plot_figs
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
parser.add_argument('--freeze', type=str, default='', help='emb = freeze embeddings, tf = freeze transformer, else empty str')
args = parser.parse_args()

task_exp = args.task_exp
img_size = args.img_size
run_name = f'{args.run_name}is{img_size}t{task_exp}'
n_tasks = 2**task_exp # these will be the num of seeds we send into get_transformed_batch
batch_size = args.batch_size
seq_len = args.seq_len #this is what we previously called "batch_size"
steps = args.train_steps
lr = args.lr
datadir = args.datadir
path = args.path
figdir = f'{path}figures/'
freeze = args.freeze
momentum = 0.9

# only train on 0s and 1s
trainloader, testloader = dataset_utils.load_mnist_01(img_size, seq_len=seq_len)
total_steps = int(np.ceil(steps/len(trainloader)))*len(trainloader)
device = 'cuda'

#instantiate model
# block size is the seq_len * 2 to account for image label interleaving
# reduce num of classes from 10 to 2 for just 0s and 1s
model = transformer.ImageICLTransformer(d_model=64, n_layer=4, device='cuda', block_size=2*seq_len, num_classes=2)

# transfer tf, (opt: freeze the tf), reset all other weights in model to random, and then retrain with 
# (frozen) transferred tf
# test on unseen, how does plateau change?
# plot test curve as well as train curve and test acc

# load trained model
# model = torch.load(f"{datadir}gradnormis{img_size}t{task_exp}.pt")
model = torch.load(f"{datadir}new2classis{img_size}t{task_exp}.pt")

for name, param in model.named_parameters():
    print(name, param)

# Save the transformer weights
transformer_state_dict = model.transformer.state_dict()

# Reinitialize the model
model = transformer.ImageICLTransformer(d_model=64, n_layer=4, device='cuda', block_size=2*seq_len, num_classes=2)

# Load the saved transformer weights back into the model
model.transformer.load_state_dict(transformer_state_dict)
model = model.to(device)

if freeze == 'tf':
    # freeze only the transformer
    for name, param in model.transformer.named_parameters():
        param.requires_grad = False
        print(name, param.requires_grad)

# add the params back to optimizer
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])

for name, param in model.named_parameters():
    print(name, param)

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

def eval_test(model, testlosses, accuracies, test_epochs, testloader, device, n_tasks, test_task_list, batch_size, criterion):
    model.eval()
    with torch.no_grad():
        testloss = 0
        num_batch = len(testloader)
        accuracy = 0
        for test_epoch in range(1, test_epochs + 1):
            correct = 0
            for images, labels in testloader:
                temp_images, temp_labels, test_task_list = dataset_utils.make_unseen_batch(images, labels, device, n_tasks, test_task_list, batch_size, 2)
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
    return accuracy, correct

losses = []
testlosses = []
testbatchlosses = []
accuracies = []
task_list = []
test_task_list = []
optimizer.zero_grad()
epochs = int(np.ceil(total_steps/len(trainloader)))
test_epochs = 5

# training loop
for epoch in (pbar := tqdm(range(1, epochs + 1))):
    model.train()
    for images, labels in trainloader:
        temp_images, temp_labels, task_list = dataset_utils.make_batch(images, labels, device, n_tasks, task_list, batch_size, 2)
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
        # if first epoch, calc the test loss and acc at each step
        if epoch == 1:
            eval_test(model, testlosses, accuracies, test_epochs, testloader, device, n_tasks, test_task_list, batch_size, criterion)
             
    checkpoint(model, epoch, optimizer, loss.item(), losses, testlosses, accuracies, run_name)
    # compute the test/val loss
    if epoch != 1:
        accuracy, correct = eval_test(model, testlosses, accuracies, test_epochs, testloader, device, n_tasks, test_task_list, batch_size, criterion)

print(correct.item())
print(accuracy)

torch.save(model, f"{datadir}{run_name}.pt")

losses = np.array(losses)
np.save(f'{datadir}losses_{run_name}', losses)
testlosses = np.array(testlosses)
np.save(f'{datadir}testlosses_{run_name}', testlosses)
accuracies = np.array(accuracies)
np.save(f'{datadir}accuracies_{run_name}', accuracies)

plot_figs.plot_loss_acc(trainloader, losses, testlosses, accuracies, figdir, run_name)