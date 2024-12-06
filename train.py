from torch import optim
import torch.nn as nn
import dataset
from torch.utils.data import DataLoader, random_split
import main
from models import Baseline
import torch

D_cost = nn.L1Loss(reduction="none")
device = main.get_device()
batch_size = 32
data = dataset.WallDataset(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
    )
train_size = int(0.6 * len(data))
val_size = len(data) - train_size

_, rem_data = random_split(data, [train_size, val_size])
train_size = int(0.8 * len(rem_data))
val_size = len(rem_data) - train_size

train_dataset, val_dataset = random_split(rem_data, [train_size, val_size])
print(len(train_dataset), len(val_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

epochs = 10
model = Baseline(device=device).to(device)
optim = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for idx, d in enumerate(train_dataloader):
        pred_embed, actual_embed = model(states=d.states,actions=d.actions,train=True)
        loss = D_cost(pred_embed,actual_embed).sum(1).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
    print(f'Loss: {total_loss/(idx+1)} Epoch: {epoch}')
    
total_loss = 0
model.eval()
for idx, d in enumerate(val_dataloader):
    with torch.no_grad():
        pred_embed, actual_embed = model(states=d.states,actions=d.actions,train=False)
        loss = D_cost(pred_embed,actual_embed).sum(1).mean()
        total_loss += loss.item()
print(f'Loss: {total_loss/(idx+1)} Validation Loss')
        
        
