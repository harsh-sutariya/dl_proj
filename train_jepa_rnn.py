from torch import optim
import torch.nn as nn
import dataset
from torch.utils.data import DataLoader, random_split
import main
import torch
import argparse
import wandb
import utils
from tqdm import tqdm
import time
import os
import models

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def compute_bt_loss(embed1, embed2, device, lam):
    """
    e1: (B, T-1, D)
    e2: (B, T-1, D)
    """
    B, T, D = embed1.shape
    normalized_embed1 = (embed1 - embed1.mean(0))/embed1.std(0) # B, T-1, D
    normalized_embed2 = (embed2 - embed2.mean(0))/embed2.std(0) # B, T-1, D
    
    corr_matrix = torch.bmm(normalized_embed1.permute(1,2,0),normalized_embed2.permute(1,0,2))/B # T-1, D, D
    c_diff = (corr_matrix - torch.eye(D).reshape(1,D, D).repeat(T,1,1).to(device)).pow(2)
    off_diagonal = (torch.ones((D, D))-torch.eye(D)).reshape(1,D,D).repeat(T,1,1).to(device)
    c_diff *= (off_diagonal*lam + torch.eye(D).reshape(1,D, D).repeat(T,1,1).to(device))
    return c_diff.sum()

def train(jepa, train_dataloader, optimizer, args, device):
    jepa.train()
    total_loss = 0
    for idx, d in tqdm(enumerate(train_dataloader)):
        pred_embed = jepa(states=d.states,actions=d.actions)
        B,T,C,H,W = d.states.shape
        actual_embed = jepa.context_encoder(d.states.reshape(-1,C,H,W)).reshape(B,T,-1)
        loss = compute_bt_loss(pred_embed[:,1:],actual_embed[:,1:],device, args.lam)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        wandb.log({"train_batch_loss":loss})
    return total_loss/(idx+1)
    

def val(jepa, val_dataloader, device):
    jepa.eval()
    total_loss = 0
    with torch.no_grad():
        for idx, d in tqdm(enumerate(val_dataloader)):
            pred_embed = jepa(states=d.states,actions=d.actions)
            B,T,C,H,W = d.states.shape
            actual_embed = jepa.context_encoder(d.states.reshape(-1,C,H,W)).reshape(B,T,-1)
            loss = compute_bt_loss(pred_embed[:,1:],actual_embed[:,1:],device, args.lam)
            total_loss += loss.item()
            wandb.log({"val_batch_loss":loss})
        return total_loss/(idx+1)
def main(args):
    wandb.login()
    with wandb.init(project="jepa-gru",entity="dl-nyu", config=vars(args)):
        start_time = time.time()
        device = get_device()
        utils.seed_numpy(args.seed)
        utils.seed_torch(args.seed)
        save_dir = os.path.join("../jepa_models_gru",str(start_time))
        print(f'Saving Dir: {save_dir}')
        os.makedirs(save_dir)
        data = dataset.WallDataset(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,)
        train_size = int(0.9 * len(data))
        val_size = len(data) - train_size
        train_dataset, val_dataset = random_split(data, [train_size, val_size])
        print(len(train_dataset), len(val_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False)
        jepa = models.JEPA_RNNCell(encoder=args.encoder, rnn=args.rnn, device=device)
        optimizer = optim.Adam(jepa.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,1e-5)
        best_val_loss = float('inf')
        for epoch in tqdm(range(args.epochs)):
            train_loss = train(jepa,train_dataloader,optimizer, args, device)
            print(f'Train Loss: {train_loss} Epoch: {epoch}')
            wandb.log({"train_loss":train_loss})
            val_loss = val(jepa,val_dataloader,device)
            print(f'Val Loss: {val_loss} Epoch: {epoch}')
            wandb.log({"val_loss":val_loss})
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                torch.save(jepa.state_dict(),
                        f'{save_dir}/{args.encoder}_{args.batch_size}_{args.lr}_best_model.pth')
            torch.save(jepa.state_dict(),
                       f'{save_dir}/{args.encoder}_{args.batch_size}_{args.lr}_epoch_{epoch}.pth')
            scheduler.step()
                    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=256)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--weight_decay",type=float,default=1e-5)
    parser.add_argument("--lam", type=float,default=0.005)
    parser.add_argument("--encoder",type=str,default="resnet")
    parser.add_argument("--seed", type=int,default=0)
    parser.add_argument("--num_workers", type=int, default=6)
    args = parser.parse_args()
    print(args)
    main(args)
