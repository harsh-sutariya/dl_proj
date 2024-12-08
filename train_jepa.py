from torch import optim
import torch.nn as nn
import dataset
from torch.utils.data import DataLoader, random_split
import main
from models import Baseline
import torch
import argparse
import wandb
import utils
from tqdm import tqdm
import time
import os
import models

D_cost = nn.MSELoss(reduction="none")
def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def main(args):
    wandb.login()
    with wandb.init(project="jepa",entity="dl-nyu", config=vars(args)):
        start_time = time.time()
        device = get_device()
        utils.seed_numpy(args.seed)
        utils.seed_torch(args.seed)
        save_dir = os.path.join("../jepa_models",str(start_time))
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
        jepa = models.JEPA(encoder=args.encoder,device=device, context_encoder_path=args.ctxt_enc_path, freeze_encoder=True)
        optimizer = optim.Adam(jepa.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        target_encoder_path = args.trg_enc_path if args.trg_enc_path  else args.ctxt_enc_path
        target_encoder = models.Encoder(args.encoder)
        target_encoder.load_state_dict(torch.load(target_encoder_path, map_location=device, weights_only=True))
        target_encoder.to(device)
        best_val_loss = float('inf')
        for epoch in tqdm(range(args.epochs)):
            jepa.train()
            total_loss = 0
            for idx, d in tqdm(enumerate(train_dataloader)):
                pred_embed = jepa(states=d.states,actions=d.actions,train=True)
                # processed_state = utils.preprocess_state(d.states)
                B,T,C,H,W = d.states.shape
                actual_embed = target_encoder(d.states.reshape(-1,C,H,W)).reshape(B,T,-1)
                loss = D_cost(pred_embed[:,1:],actual_embed[:,1:]).sum(1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Train Loss: {total_loss/(idx+1)} Epoch: {epoch}')
            wandb.log({"train_loss":total_loss/(idx+1)})
            jepa.eval()
            total_loss = 0
            with torch.no_grad():
                for idx, d in tqdm(enumerate(val_dataloader)):
                    pred_embed = jepa(states=d.states,actions=d.actions)
                    # d.states = utils.preprocess_state(d.states)
                    B,T,C,H,W = d.states.shape
                    actual_embed = target_encoder(d.states.reshape(-1,C,H,W)).reshape(B,T,-1)
                    loss = D_cost(pred_embed[:,1:],actual_embed[:,1:]).sum(1).mean()
                    total_loss += loss.item()
                print(f'Val Loss: {total_loss/(idx+1)} Epoch: {epoch}')
                wandb.log({"val_loss":total_loss/(idx+1)})
                if best_val_loss > total_loss/(idx+1):
                    best_val_loss = total_loss/(idx+1)
                    torch.save(jepa.state_dict(),
                            f'{save_dir}/{args.encoder}_{args.batch_size}_{args.lr}_best_model.pth')
            torch.save(jepa.state_dict(),
                       f'{save_dir}/{args.encoder}_{args.batch_size}_{args.lr}_epoch_{epoch}.pth')
                    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=256)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--weight_decay",type=float,default=1e-5)
    parser.add_argument("--lam", type=float,default=5e-3)
    parser.add_argument("--encoder",type=str,default="resnet50")
    parser.add_argument("--seed", type=int,default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ctxt_enc_path",type=str,required=True)
    parser.add_argument("--trg_enc_path",type=str,default=None)
    args = parser.parse_args()
    print(args)
    main(args)
