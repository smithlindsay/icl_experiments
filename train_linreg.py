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
import torch.optim as optim
from torchvision import models
import scipy.linalg as linalg
import torch.autograd as autograd
import inspect


def FIM(model, device, data, b_stop=10):
    grad_idxs = np.array(list(range(0,99,2)))
    P = sum(p.numel() for p in model.parameters())



    model.eval()
    idx = 0
    params = [param for param in model.parameters()]

    xs, ys = data
    xs,ys = autograd.Variable(xs).to(device), autograd.Variable(ys).to(device)
    output = model((xs,ys))
    B, T, _ = output.shape

    N = len(grad_idxs) * (b_stop + 1)
    jac = np.empty((N, P))
    for b in range(B):
        if b > b_stop:
            break
        for j,g_idx in enumerate(grad_idxs):
            grads = autograd.grad(output[b,g_idx,:], params, retain_graph=True,materialize_grads=True) # check by hand this gives the right gradient
            grads = torch.cat([g.flatten() for g in grads])
            jac[idx] = grads.cpu().detach()
            idx += 1
    jac = jac / np.sqrt(N)

    sigma = linalg.svd(jac,full_matrices=False,compute_uv=False)
    return sigma ** 2

#compute test loss on new w's
def test_model(model, device, grad_idxs, criterion, epoch, dim, batch_size,
               batches_per_epoch, seq_len, noise_std, offset=1000,test_batches=50,
               ws=None):
    total_loss=0
    for b in range(test_batches):
        model.eval()
        xs, ys, ws = dataset_utils.gen_linreg_data(b+offset*batches_per_epoch*epoch,
                                                   batch_size=batch_size,dim=dim,
                                                   n_samples=seq_len,device=device,
                                                   noise_std=noise_std,
                                                   ws=ws)

        outputs = model((xs,ys))
        pred = outputs[:,grad_idxs].squeeze()
        true_ys = ys[:,:,0].squeeze()
        loss = criterion(pred,true_ys)

        total_loss += loss.item()

    return total_loss/test_batches

def test_cone_falloff(model, device, grad_idxs, criterion, epoch, dim, batch_size,
               batches_per_epoch, seq_len, noise_std, offset=1000,test_batches=10,
               start_angle=0, end_angle=180, strip_width=5, **kwargs):
    angles = []
    losses = []
    for a in range(start_angle, end_angle, strip_width):
        a *= np.pi/180
        ws = dataset_utils.sample_cone(batch_size, dim, max_theta=a+strip_width, min_theta=a)
        loss = test_model(model, device, grad_idxs, criterion, epoch, dim, batch_size,
               batches_per_epoch, seq_len, noise_std, offset=1000,test_batches=50,
               ws=ws)
        angles.append(a)
        losses.append(loss)
    return angles, losses

def get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs

def train(batch_size=128, lr=3e-4, epochs=120, batches_per_epoch=100, device='cuda',
          seq_len=50, num_workers=0, d_model=128, n_layer=10, dim=10, noise_std=1, outdir="outputs/", 
          switch_epoch=-1, pretrain_size=2**10, angle=180):

    kwargs = get_kwargs()

    angle = angle * np.pi/180

    Path(outdir).mkdir(parents=True, exist_ok=True)

    print(num_workers, " workers")
    print("angle: ", angle)

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
    traces = []

    ws = torch.randn(pretrain_size, dim, device=device)

    for epoch in range(1,epochs+1):
        print("Epoch: ", epoch)
        t1 = time.time()

        model.train()
        for b in range(batches_per_epoch):
            if epoch <= switch_epoch:
                #batch_ws_idxs = np.random.permutation(pretrain_size)[:batch_size]
                batch_ws = dataset_utils.sample_cone(batch_size, dim, angle)#ws[batch_ws_idxs]
                xs, ys, _ = dataset_utils.gen_linreg_data(b+batches_per_epoch*epoch,batch_size=batch_size,
                                                      dim=dim,n_samples=seq_len,device=device,noise_std=noise_std,
                                                      ws=batch_ws)
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
        test_loss = test_model(model, device, grad_idxs, criterion, epoch, dim, batch_size,
                               batches_per_epoch, seq_len, noise_std, offset=1000*epoch)
        test_loss_history[epoch-1] = test_loss
        if False and (epoch - 1) % 3 == 0:
            eigs = FIM(model,'cuda',(xs,ys),b_stop=10)
            np.save(outdir+"eigs{0}.npy".format(epoch), eigs)
            eigs_all.append(eigs)
            eigs_xs.append(epoch)
            traces.append(eigs.sum())
            

        t2 = time.time()
        print("Epoch complete, time:", (t2-t1)/60, "minutes, loss: {0}, test loss: {1}".format(loss.item(),test_loss))

    loss_history = np.array(loss_history)
    plt.figure()
    plt.plot(loss_history, label="Train loss (cone angle = {0})".format(angle))
    x_epoch = np.arange(len(test_loss_history))*batches_per_epoch
    plt.plot(x_epoch,test_loss_history,label="Test loss")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.savefig(outdir + "loss_history{0}.png".format(angle))
    np.save(outdir + "loss_history.npy", loss_history)

    torch.save(model.state_dict(), outdir + "final.th")

    """traces = np.array(traces)
    x_traces = np.arange(len(traces))*batches_per_epoch
    x_epoch = np.arange(len(test_loss_history))*batches_per_epoch
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(loss_history,label="Train loss")
    ax1.plot(x_epoch,test_loss_history,label="Test loss")
    ax2.scatter(3*x_traces,traces,color='green')
    ax2.set_yscale('log')
    ax1.set_ylabel('MSE (blue & orange)')
    ax2.set_ylabel('Tr F (green)')
    ax1.set_xlabel('step')

    plt.savefig(outdir + 'loss_history_fisher.png')"""
    angles, losses = test_cone_falloff(model, device, grad_idxs, criterion, epoch, dim, batch_size,
               batches_per_epoch, seq_len, noise_std, offset=1000,test_batches=10,
               start_angle=0, end_angle=180, strip_width=5)
    plt.figure()
    plt.plot(angles, losses)
    return model, kwargs
