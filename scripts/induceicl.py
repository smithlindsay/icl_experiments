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

parser = argparse.ArgumentParser(description='take embeddings from trained model exhibiting ICL, put frozen emb on trained model not ICL')
parser.add_argument('--task_exp', type=int, default=16, help='exp of 2**___: number of tasks to augment the dataset with')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--seq_len', type=int, default=100, help='sequence length')
parser.add_argument('--train_steps', type=int, default=30_000, help='min number of training steps')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--datadir', type=str, default='/scratch/gpfs/ls1546/icl_experiments/data/')
parser.add_argument('--path', type=str, default='/scratch/gpfs/ls1546/icl_experiments/', help='path to icl dir')
parser.add_argument('--img_size', type=int, default=28, help='size of the MNIST image')
parser.add_argument('--random_init_tr', type=str, default='', help='True = random init transformer, else empty str')
parser.add_argument('--freeze', type=str, default='', help='emb = freeze embeddings, tr = freeze transformer, else empty str')
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
freeze = args.freeze
if args.random_init_tr == 'True':
    random_init_tr = True
else:
    random_init_tr = False

trainloader, testloader = dataset_utils.load_mnist_01(img_size, seq_len=seq_len)
total_steps = int(np.ceil(steps/len(trainloader)))*len(trainloader)
device = 'cuda'

#instantiate model
# block size is the seq_len * 2 to account for image label interleaving
# reduce num of classes from 10 to 2 for just 0s and 1s
model = transformer.ImageICLTransformer(d_model=64, n_layer=4, device='cuda', block_size=2*seq_len, num_classes=2)

# test on unseen tasks to see if ICL occurs

# load a model that doesn't exhibit ICL, load a model that does exhibit ICL and put those embeddings on the model
#  that doesn't exhibit ICL
# freeze embeddings, train and then test on unseen, the plateau in training should be gone
# plot test curve as well as train curve

# load trained model w/ ICL
model = torch.load(f"{datadir}gradnormis16t16.pt")

for name, param in model.named_parameters():
    print(name, param)

# Save the embedding weights
embed_state_dict = model.img_embed.state_dict()
label_embed_state_dict = model.label_embed.state_dict()

# Reinitialize the model
model = transformer.ImageICLTransformer(d_model=64, n_layer=4, device='cuda', block_size=2*seq_len, num_classes=2)

if not random_init_tr:
    # load the transformer from the model that doesn't exhibit ICL
    model = torch.load(f"{datadir}gradnormis16t12.pt")

# Load the saved embedding weights back into the model
model.img_embed.load_state_dict(embed_state_dict)
model.label_embed.load_state_dict(label_embed_state_dict)
model = model.to(device)

if freeze == 'emb':
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

losses = []
testlosses = []
testbatchlosses = []
task_list = []
test_task_list = []
accuracies = []
optimizer.zero_grad()
epochs = int(np.ceil(total_steps/len(trainloader)))

# training loop
for epoch in (pbar := tqdm(range(1, epochs + 1))):
    model.train()
    for images, labels in trainloader:
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
    checkpoint(model, epoch, optimizer, loss.item(), run_name)
    # compute the test/val loss
    model.eval()
    with torch.no_grad():
        correct = 0
        testloss = 0
        num_batch = len(testloader)
        for images, labels in testloader:
            temp_images, temp_labels, test_task_list = dataset_utils.make_unseen_batch(images, labels, device, n_tasks, test_task_list, batch_size)
            outputs = model((temp_images,temp_labels))
            pred = outputs[:,-1,:]
            batchloss = criterion(pred,temp_labels[:,-1])
            _, predicted = torch.max(pred, 1)
            correct += (predicted == temp_labels[:,-1]).sum()
            testloss += batchloss.item()
            testbatchlosses.append(batchloss.item())
        accuracy = 100 * (correct.item()) / (batch_size*num_batch)
        accuracies.append(accuracy)
        testloss /= len(testloader)
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

# make x values for number of steps per epoch to plot test loss at end of epoch on the same plot as train batch loss
x = np.arange(len(trainloader), len(trainloader)*epochs+1, len(trainloader))

title_str = ''
if random_init_tr:
    title_str = ', random init tr'
else:
    title_str = ', tr tasks:2**12'
suptitle_str = ''
if freeze == 'emb':
    suptitle_str = ', froz emb'
else:
    suptitle_str = ', no freeze'


plt.figure()
plt.plot(losses, label='train')
plt.plot(x, testlosses, zorder=5, label='test')
plt.plot(x, accuracies, zorder=10, label='test accuracy', color='black', ls='--')
plt.title(f'loss - img size:({img_size}x{img_size}), emb tasks:2**16{title_str}')
plt.suptitle(f'train/test on 2**{task_exp} tasks{suptitle_str}')
plt.xlabel('batch')
plt.ylabel('loss')
plt.legend()
plt.savefig(f'{figdir}loss{run_name}.pdf')