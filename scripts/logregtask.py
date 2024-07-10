import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/scratch/gpfs/ls1546/icl_experiments/')
import transformer
import dataset_utils
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cuda'
batch_size = 64
lr = 3e-4
momentum = 0.9
epochs = 60
batches_per_epoch = 100
# device = 'cuda'
seq_len = 50
d_model = 64
n_layer = 10
dim=5

model = transformer.LinRegTransformer(d_model=d_model,device=device,
                                        block_size=seq_len*2,n_layer=n_layer,input_dim=dim)

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = torch.nn.BCEWithLogitsLoss()
grad_idxs = np.arange(0,99,2)

loss_history = [0]*batches_per_epoch*epochs

for epoch in (pbar := tqdm(range(1, epochs + 1))):
    for b in range(batches_per_epoch):
        model.train()

        xs, ys, _ = dataset_utils.gen_logreg_data(b+batches_per_epoch*epoch,batch_size=batch_size,dim=dim,n_samples=seq_len,device=device)

        optimizer.zero_grad()
        outputs = model((xs,ys))
        pred = outputs[:,grad_idxs].squeeze()
        true_ys = ys[:,:,0].squeeze()
        loss = criterion(pred,true_ys)
        loss.backward()
        optimizer.step()
        loss_history[b+batches_per_epoch*(epoch-1)] = loss.item()
    pbar.set_description(f"epoch: {epoch}, loss: {loss.item():4f}")

loss_history = np.array(loss_history)
plt.plot(loss_history)
plt.xlabel("Step")
plt.ylabel("Binary Cross Entropy Loss")
plt.savefig('logreg_bceloss.pdf')

torch.save(model.state_dict(), "finalbce.th")

#compute test loss on new w's
offset = 1000
test_batches = 10
num_corr = 0
total_examples = 0
for b in range(test_batches):
    model.eval()
    xs, ys, ws = dataset_utils.gen_logreg_data(b+offset*batches_per_epoch*epoch,batch_size=batch_size,dim=dim,n_samples=seq_len,device=device)

    outputs = model((xs,ys))
    pred = outputs[:,grad_idxs].squeeze()
    true_ys = ys[:,:,0].squeeze()
    loss = criterion(pred,true_ys)

    num_corr += torch.sum(torch.round(pred) == torch.round(true_ys))
    total_examples += pred.shape[0]

print('test accuracy: ', num_corr/total_examples)

#look at some of the model outputs to check
xs, ys, _ = dataset_utils.gen_logreg_data(offset*batches_per_epoch*epoch,batch_size=batch_size,dim=dim,n_samples=seq_len,device=device)

model.eval()
outputs = model((xs,ys))
pred = outputs[:,grad_idxs].squeeze()
true_ys = ys[:,:,0].squeeze()

print("test sequence 0 last 5")
print('pred ', pred[0,-5:])
print('true ', true_ys[0,-5:])

print("test sequence 10 last 5")
print('pred ', pred[10,:5])
print('true ', true_ys[10,:5])

print("sigmoid test sequence 0 last 5")
print('pred ', F.sigmoid(pred[0,-5:]))
print('true ', true_ys[0,-5:])

print("sigmoid test sequence 10 last 5")
print('pred ', F.sigmoid(pred[10,:5]))
print('true ', true_ys[10,:5])

criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
losses = criterion(pred,true_ys)
print(losses.shape)
avgs = torch.mean(losses,axis=0)
plt.plot(avgs.cpu().detach())
plt.ylabel("bce loss")
plt.xlabel("# examples in context")