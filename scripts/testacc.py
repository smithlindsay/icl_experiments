import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
import argparse
sys.path.insert(1, '/scratch/gpfs/ls1546/icl_experiments/')
import transformer
import dataset_utils
from tqdm.auto import tqdm

RUN_MAPPING = dict(
    orig12='2class16is16t12',
    orig16='2class16is16t16',
    emb='16to12nofreezerandis16t12',
    embfreeze='16to12freezerandis16t12',
    tf='tfnofreezerandis16t12',
    tffreeze='tffreezerandis16t12'
)

parser = argparse.ArgumentParser(description='calc test accuracy')
parser.add_argument("--run", type=str, choices=list(RUN_MAPPING.keys()))
parser.add_argument("--shuffled", action='store_true', help='Enable dataloader to be shuffled')
args = parser.parse_args()

run_name = RUN_MAPPING[args.run]
datadir = '/scratch/gpfs/ls1546/icl_experiments/data/'
figdir = '/scratch/gpfs/ls1546/icl_experiments/figures/'
img_size = 16
seq_len = 100
device = 'cuda'
batch_size = 32
if args.run == 'orig16':
    n_tasks = 2**16
else:
    n_tasks = 2**12
if args.shuffled:
    shuffled = True
    dltype = 'shuf'
else:
    shuffled = False
    dltype = 'unshuf'

print(args.run)

# calc test acc on unseen tasks by going through many epochs of test data
test_epochs = 100
criterion = torch.nn.CrossEntropyLoss()

# load model
model = transformer.ImageICLTransformer(d_model=64, n_layer=4, device='cuda', block_size=2*seq_len, num_classes=2)
model = torch.load(f"{datadir}{run_name}.pt")
model = model.to(device)
model.eval()
for num in range(50):
    # load testloader
    # only train on 0s and 1s
    _, testloader = dataset_utils.load_mnist_01(img_size, seq_len=seq_len, shuffled=shuffled)
    
    # test loop
    # record test acc at end of each epoch and final test acc
    def eval_test(model, testlosses, accuracies, test_epochs, testloader, device, n_tasks, test_task_list, batch_size, criterion):
        with torch.no_grad():
            testloss = 0
            num_batch = len(testloader)
            for test_epoch in tqdm(range(1, test_epochs + 1)):
                correct = 0
                accuracy = 0
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
            testloss /= (len(testloader) * test_epochs)
            testlosses.append(testloss)
        return accuracy, correct
    
    testlosses = []
    testbatchlosses = []
    accuracies = []
    test_task_list = []
    
    accuracy, correct = eval_test(model, testlosses, accuracies, test_epochs, testloader, device, n_tasks, test_task_list, batch_size, criterion)
    
    testlosses = np.array(testlosses)
    np.save(f'{datadir}finaltestlosses_{args.run}_{dltype}{num}', testlosses)
    accuracies = np.array(accuracies)
    np.save(f'{datadir}finalaccuracies_{args.run}_{dltype}{num}', accuracies)
    print(shuffled)
    print(f'correct: {correct.item()}')
    print(f'accuracy: {accuracies.mean()}')
