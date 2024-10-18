# figure plotting helper functions
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
sys.path.insert(1, '/scratch/gpfs/ls1546/icl_experiments/')
import transformer
import dataset_utils


# plot of train and test loss and test accuracies
def plot_loss_acc(trainloader, losses, testlosses, accuracies, figdir, run_name):
    # make x values for number of steps per epoch to plot test loss at end of epoch on the same plot as train batch loss
    steps_per_epoch = len(trainloader)
    x_initial = np.arange(1, steps_per_epoch+1)
    x_remaining = np.arange(steps_per_epoch*2, steps_per_epoch+1 + (len(accuracies) - steps_per_epoch)*steps_per_epoch, steps_per_epoch)
    x = np.concatenate((x_initial, x_remaining))
    
    plt.figure()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(losses, label='train')
    ax1.plot(x, testlosses, zorder=5, label='test')
    ax1_twin = ax1.twinx()
    color = 'tab:red'
    ax1_twin.plot(x, accuracies, label='test acc', color=color)
    ax1_twin.tick_params(axis='y', labelcolor=color)
    
    ax1.set_xlabel('batch')
    ax1.set_ylabel('loss')
    ax1_twin.set_ylabel('test accuracy', color=color)
    # set twin y axis scale to 0 - 100
    ax1_twin.set_ylim(0, 100)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    
    ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2)
    plt.title(f'losses and test acc for {run_name}')
    plt.tight_layout()
    plt.savefig(f'{figdir}loss{run_name}_acc.pdf')