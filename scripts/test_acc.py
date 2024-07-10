import torch
import numpy as np
import sys
import argparse
sys.path.insert(1, '/scratch/gpfs/ls1546/icl_experiments/')
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import dataset_utils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='test a transformer model on 2 classes of MNIST for ICL')
parser.add_argument('--task_exp', type=int, default=16, help='exp of 2**___: number of tasks to augment the dataset with')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--seq_len', type=int, default=100, help='sequence length')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--datadir', type=str, default='/scratch/gpfs/ls1546/icl_experiments/data/')
parser.add_argument('--img_size', type=int, default=28, help='size of the MNIST image')
args = parser.parse_args()

task_exp = args.task_exp
run_name = f'{args.run_name}imgsize{args.img_size}tasks{task_exp}'
# these will be the num of seeds we send into get_transformed_batch
n_tasks = 2**task_exp
batch_size = args.batch_size
#this is what we previously called "batch_size"
seq_len = args.seq_len

datadir = args.datadir

img_size = args.img_size

#load MNIST
transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

test_data = datasets.MNIST('../data', train=False, transform=transform)

device = 'cuda'

run_name = 'all_tok_model_mnistheatmap'
#instantiate model
model = torch.load(f"{datadir}all_tok_model_{run_name}frozemb.pth")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

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
    if not seen_tasks:
        run_name = f"{run_name}unseen"

    for images, labels in tqdm(testloader):
        # add batch dimension
        images = torch.unsqueeze(images, 0).to(device)
        labels = torch.unsqueeze(labels, 0).to(device)
        if seen_tasks:
            temp_images, temp_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, seed=np.random.randint(n_tasks))
        else:
            temp_images, temp_labels, curr_seed = dataset_utils.get_transformed_batch(images, labels, seed=np.random.randint(n_tasks)+n_tasks)
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


