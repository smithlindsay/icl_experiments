import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from einops import rearrange
import argparse
sys.path.insert(1, '/scratch/gpfs/ls1546/icl_experiments/')
import transformer
import dataset_utils
import plot_figs
from tqdm.auto import tqdm
import pickle

parser = argparse.ArgumentParser(description='Train a transformer model on 3 classes of MNIST for ICL')
parser.add_argument('--task_exp', type=int, default=16, help='exp of 2**___: number of tasks to augment the dataset with')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--seq_len', type=int, default=100, help='sequence length')
parser.add_argument('--train_steps', type=int, default=30_000, help='min number of training steps')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--datadir', type=str, default='/scratch/gpfs/ls1546/icl_experiments/data/')
parser.add_argument('--path', type=str, default='/scratch/gpfs/ls1546/icl_experiments/', help='path to icl dir')
parser.add_argument('--img_size', type=int, default=28, help='size of the MNIST image')
parser.add_argument('--num_classes', type=int, default=3, help='number of classes to train on')
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
num_classes = args.num_classes

if num_classes == 3:
    trainloader, testloader = dataset_utils.load_mnist_3classes(img_size, seq_len=seq_len)
elif num_classes == 5:
    trainloader, testloader = dataset_utils.load_mnist_5classes(img_size, seq_len=seq_len)
else:
    trainloader, testloader = dataset_utils.load_mnist_01(img_size, seq_len=seq_len)
total_steps = int(np.ceil(steps/len(trainloader)))*len(trainloader)
device = 'cuda'


#instantiate model
# block size is the seq_len * 2 to account for image label interleaving
# reduce num of classes from 10 to 2 for just 0s and 1s
model = transformer.ImageICLTransformer(d_model=64, n_layer=4, device='cuda', block_size=2*seq_len, num_classes=num_classes)

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
avg_batch_losses = []
optimizer.zero_grad()
epochs = int(np.ceil(total_steps/len(trainloader)))

# training loop
for epoch in (pbar := tqdm(range(1, epochs + 1))):
    for images, labels in trainloader:
        model.train()
        temp_images, temp_labels, task_list = dataset_utils.make_batch(images, labels, device, n_tasks, task_list, batch_size, num_classes)
        outputs = model((temp_images,temp_labels))
        pred = outputs[:, 0::2, :]
        del outputs
        # train by predicting on all tokens, not just the last one
        pred = rearrange(pred, 'b t ... -> (b t) ...')
        labels = rearrange(temp_labels, 'b t -> (b t)')
        crit = torch.nn.CrossEntropyLoss(reduction='none')
        # save losses for each batch for loss vs. num examples
        all_losses = rearrange(crit(pred, labels), "(b t) -> b t", b=batch_size).cpu().detach().numpy()
        avg_batch_losses.append(np.mean(all_losses, axis=0))

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
# np.save(f'{datadir}avg_batch_losses_{run_name}', avg_batch_losses)
# save avg_batch_losses as pickle
with open(f'{datadir}avg_batch_losses_{run_name}.pkl', 'wb') as f:
    pickle.dump(avg_batch_losses, f)

plot_figs.plot_loss_acc(trainloader, losses, testlosses, accuracies, figdir, run_name)