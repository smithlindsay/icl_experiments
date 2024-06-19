import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import embedding
import transformer
import dataset_utils
import torch.backends.cudnn as cudnn
import argparse
import time
from pathlib import Path

#compute test loss on new w's
def test_model(model, grad_idxs, criterion, epoch, offset=1000,test_batches=50):
    total_loss=0
    for b in range(test_batches):
        model.eval()
        xs, ys, ws = dataset_utils.gen_linreg_data(b+offset*batches_per_epoch*epoch,
                                                   batch_size=batch_size,dim=dim,
                                                   n_samples=seq_len,device=device,
                                                   noise_std=noise_std)

        outputs = model((xs,ys))
        pred = outputs[:,grad_idxs].squeeze()
        true_ys = ys[:,:,0].squeeze()
        loss = criterion(pred,true_ys)

        total_loss += loss.item()

    return total_loss/test_batches

batch_size = 128
lr = 3e-4
epochs = 120
batches_per_epoch = 100
device = 'cuda'
seq_len = 50
num_workers = 0
d_model = 128
n_layer = 10
dim=10
noise_std = 1
outdir = "outputs/"

def train(batch_size=128, lr=3e-4, epochs=120, batches_per_epoch=100, device='cuda',
          seq_len=50, num_workers=0, d_model=128, n_layer=10, dim=10, noise_std=1, outdir="outputs/", 
          switch_epoch=-1):

    Path(outdir).mkdir(parents=True, exist_ok=True)

    print(num_workers, " workers")

    model = transformer.LinRegTransformer(d_model=d_model,device=device,
                                            block_size=seq_len*2,n_layer=n_layer,input_dim=dim)

    model = model.to(device)
    P = sum([p.numel() for p in model.parameters()])
    print(P)

    

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    grad_idxs = np.array(list(range(0,99,2)))

    loss_history = [0]*batches_per_epoch*epochs
    test_loss_history = [0]*epochs

    eigs_all = []
    eigs_xs = []

    ws = torch.randn(batch_size, dim, device=device)

    for epoch in range(1,epochs+1):
        print("Epoch: ", epoch)
        t1 = time.time()

        model.train()
        for b in range(batches_per_epoch):
            if epoch <= switch_epoch:
                xs, ys, _ = dataset_utils.gen_linreg_data(b+batches_per_epoch*epoch,batch_size=batch_size,
                                                      dim=dim,n_samples=seq_len,device=device,noise_std=noise_std,
                                                      ws=ws)
            else:
                xs, ys, _ = dataset_utils.gen_linreg_data(b+batches_per_epoch*epoch,batch_size=batch_size,
                                                      dim=dim,n_samples=seq_len,device=device,noise_std=noise_std,
                                                      ws=None)
            optimizer.zero_grad()
            outputs = model((xs,ys))
            pred = outputs[:,grad_idxs].squeeze()
            true_ys = ys[:,:,0].squeeze()
            loss = criterion(pred,true_ys)
            loss.backward()
            optimizer.step()
            loss_history[b+batches_per_epoch*(epoch-1)] = loss.item()
        if (epoch - 1) % 3 == 0:
            torch.save(model.state_dict(),outdir + "checkpoint{0}.th".format(epoch))
        test_loss = test_model(model, grad_idxs, criterion, epoch, offset=1000*epoch)
        test_loss_history[epoch-1] = test_loss
        """if (epoch - 1) % 3 == 0:
            eigs = FIM(model,'cuda',(xs,ys),b_stop=5)
            eigs_all.append(eigs)
            eigs_xs.append(epoch)"""

        t2 = time.time()
        print("Epoch complete, time:", (t2-t1)/60, "minutes, loss: {0}, test loss: {1}".format(loss.item(),test_loss))

    loss_history = np.array(loss_history)
    plt.plot(loss_history)
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.savefig(outdir + 'loss_history.png')

    torch.save(model.state_dict(), outdir + "final.th")
