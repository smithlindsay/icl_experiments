import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import transformer
import dataset_utils
from tqdm.auto import tqdm
import sys
from einops import rearrange
import argparse
sys.path.insert(1, '/scratch/gpfs/ls1546/icl_experiments/')

parser = argparse.ArgumentParser(description='Train a transformer model on 2 classes of MNIST for ICL')
parser.add_argument('--task_exp', type=int, default=16, help='exp of 2**___: number of tasks to augment the dataset with')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--seq_len', type=int, default=100, help='sequence length')
parser.add_argument('--train_steps', type=int, default=30_000, help='min number of training steps')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--datadir', type=str, default='/scratch/gpfs/ls1546/icl_experiments/data/')
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

total_steps = int(np.ceil(steps/len(trainloader)))*len(trainloader)

device = 'cuda'

#instantiate model
# block size is the seq_len * 2 to account for image label interleaving
# reduce num of classes from 10 to 2
model = transformer.ImageICLTransformer(d_model=64, n_layer=4, device='cuda', block_size=2*seq_len, num_classes=2)

# test on unseen tasks to see if ICL occurs
# once ICL occurs, freeze the embeddings, reset all other weights in model to random, and then retrain with frozen embeddings
# test on unseen, the plateau in training should be gone
# plot test curve as well as train curve

# load the 16x16, 2**16 task model
run_name = 'all_tok_model_mnistheatmap'
model = torch.load(f"{datadir}all_tok_model_mnistheatmapimgsize16tasks16.pth")
model = model.to(device)

for name, param in model.named_parameters():
    print(name, param)

# Save the embedding weights
embed_state_dict = model.img_embed.state_dict()
label_embed_state_dict = model.label_embed.state_dict()

# Reinitialize the model
model = transformer.ImageICLTransformer(d_model=64, n_layer=4, device='cuda', block_size=2*seq_len, num_classes=2)

# Load the saved embedding weights back into the model
model.img_embed.load_state_dict(embed_state_dict)
model.label_embed.load_state_dict(label_embed_state_dict)
model = model.to(device)

# freeze only the embeddings
for name, param in model.img_embed.named_parameters():
    param.requires_grad = False
    print(name, param.requires_grad)

for name, param in model.label_embed.named_parameters():
    param.requires_grad = False
    print(name, param.requires_grad)

# add the params back to optimizer
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])

losses = []
model.train();

for name, param in model.named_parameters():
    print(name, param)

def checkpoint(model, epoch, optimizer, loss, run_name):
    path = f"{datadir}ckpt{run_name}{epoch}.pt"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

task_list = []
optimizer.zero_grad()
epochs = int(np.ceil(total_steps/len(trainloader)))

for epoch in (pbar := tqdm(range(1, epochs + 1))):
    for images, labels in trainloader:
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

        # images.shape = batch, seq, 1, 28, 28
        # labels.shape = batch, seq
        del temp2_images, temp2_labels

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
    checkpoint(model, epoch, optimizer, loss.item(), run_name)

torch.save(model, f"{datadir}all_tok_model_{run_name}frozemb.pth")

losses = np.array(losses)
np.save(f'losses_{run_name}', losses)

plt.figure()
plt.plot(losses)
plt.title(f'batch loss - frozen embed - img size:({img_size}x{img_size}), tasks:2**{task_exp}')
plt.xlabel('batch')
plt.ylabel('loss')
plt.savefig(f'batchloss{run_name}frozemb.pdf')

# test accuracy on seen and unseen tasks

# just test on 0s and 1s from mnist
idx = (test_data.targets == 0) | (test_data.targets == 1)
test_data.targets = test_data.targets[idx]
test_data.data = test_data.data[idx]

def test_model(model, test_data, n_tasks, batch_size, seq_len, run_name, device='cuda', seen_tasks=True):
    correct = 0
    model.eval()
    
    test_task_list = []
    testlosses = []
    testloader = torch.utils.data.DataLoader(test_data, batch_size=seq_len, shuffle=False)
    num_batch = len(testloader)

    for images, labels in tqdm(testloader):
        # add batch dimension
        images = torch.unsqueeze(images, 0).to(device)
        labels = torch.unsqueeze(labels, 0).to(device)
        if seen_tasks:
            temp_images, temp_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, seed=np.random.randint(n_tasks))
        else:
            temp_images, temp_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, seed=np.random.randint(n_tasks)+n_tasks)
            run_name = f"{run_name}unseen"
        test_task_list.append(curr_seed)
        for batch in range(1, batch_size):
            if seen_tasks:
                temp2_images, temp2_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, seed=np.random.randint(n_tasks))
            else:
                temp2_images, temp2_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, seed=np.random.randint(n_tasks)+n_tasks)
            temp_images = torch.cat((temp_images, temp2_images), 0)
            temp_labels = torch.cat((temp_labels, temp2_labels), 0)
            test_task_list.append(curr_seed)

        del temp2_images, temp2_labels

        outputs = model((temp_images,temp_labels))
        pred = outputs[:,-1,:]
        loss = criterion(pred,temp_labels[:,-1])
        _, predicted = torch.max(pred, 1)
        correct += (predicted == temp_labels[:,-1]).sum()
        testlosses.append(loss.item())
    accuracy = 100 * (correct.item()) / (batch_size*num_batch)
    print(correct.item())
    print(accuracy)
    testlosses = np.array(testlosses)
    np.save(f'testlosses_{run_name}', testlosses)

test_model(model, test_data, n_tasks, batch_size, seq_len, run_name, device='cuda', seen_tasks=True)
print(f'test accuracy from {run_name} model on SEEN tasks')

test_model(model, test_data, n_tasks, batch_size, seq_len, run_name, device='cuda', seen_tasks=False)
print(f'test accuracy from {run_name} model on UNSEEN tasks')
testlosses = np.load(f'testlosses_{run_name}unseen.npy')
plt.figure()
plt.plot(testlosses)
plt.title(f'unseen test loss - frozen embed - img size:({img_size}x{img_size}), tasks:2**{task_exp}')
plt.xlabel('batch')
plt.ylabel('test loss')
plt.savefig(f'testbatchloss{run_name}frozemb.pdf')
