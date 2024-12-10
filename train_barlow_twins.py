from torch import optim
import torch.nn as nn
import dataset
from torch.utils.data import DataLoader
import models
import torch
import torchvision.transforms.v2 as v2
import argparse
from tqdm import tqdm
import wandb
import utils
import time
import os
import schedulers

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def main(args):
    wandb.login()
    with wandb.init(project="encoder", entity="dl-nyu", config=vars(args)):
        start_time = time.time()
        device = get_device()
        utils.seed_numpy(args.seed)
        utils.seed_torch(args.seed)
        save_dir = os.path.join("../saved_models",str(start_time))
        os.mkdir(save_dir)
        data = dataset.ObsDataset(data_path="/scratch/DL24FA/train")
        print("created dataset")
        dataloader = DataLoader(data,batch_size=args.batch_size,shuffle=True,pin_memory=False, num_workers=args.num_workers)
        encoder = models.Encoder(encoder=args.encoder)
        encoder = encoder.to(device)
        optimizer = optim.Adam(encoder.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,1e-5)
        # transforms  = v2.RandomChoice([v2.ColorJitter(),v2.GaussianBlur(3)])
        t1 = v2.ColorJitter()
        t2 = v2.GaussianBlur(3)
        location_transform = v2.RandomAffine(90,translate=(0.25,0.25),interpolation=v2.InterpolationMode.BILINEAR)
        off_diagonal = (torch.ones((encoder.repr_dim, encoder.repr_dim))-torch.eye(encoder.repr_dim)).to(device)
        cosine_similarity = nn.CosineSimilarity()
        best_loss = float('inf')
        wandb.watch(encoder, log="all")
        for epoch in tqdm(range(args.epochs)):
            total_loss = 0
            bt_loss = 0
            contrastive_loss = 0
            for idx, img in tqdm(enumerate(dataloader)):
                img1 = t1(img).to(device)
                img2 = t2(img).to(device)
                embed1 = encoder(img1)
                embed2 = encoder(img2)
                normalized_embed1 = (embed1 - embed1.mean(0))/embed1.std(0)
                normalized_embed2 = (embed2 - embed2.mean(0))/embed2.std(0)
                selected = embed1 if torch.rand(1) > 0.5 else embed2
                if torch.rand > 0.5:
                    img3 = location_transform(img1)
                    selected = embed1
                else:
                    img3 = location_transform(img2)
                    selected = embed2
                embed3 = encoder(img3)
                cos = nn.CosineEmbeddingLoss(margin=args.margin)(selected, embed3, torch.ones(selected.shape[0], device=device)*-1)
                corr_matrix = torch.matmul(normalized_embed1.T,normalized_embed2)/embed1.shape[0]
                c_diff = (corr_matrix - torch.eye(encoder.repr_dim).to(device)).pow(2)
                c_diff *= (off_diagonal*args.lam + torch.eye(encoder.repr_dim).to(device))
                bt_l = c_diff.sum()
                contr_l = cos.mean()
                loss = bt_l+ contr_l
                bt_loss += bt_l.item()
                contrastive_loss+= contr_l.item()
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            wandb.log({"loss":total_loss/len(dataloader), "lr": scheduler.get_last_lr()[0], "epoch":epoch, "bt_loss": bt_loss/len(dataloader), "contrastive_loss":contrastive_loss/len(dataloader)})
            if total_loss/len(dataloader) < best_loss:
                best_loss = total_loss/len(dataloader)
                torch.save(encoder.state_dict(),
                           f'{save_dir}/{args.encoder}_{args.batch_size}_{args.lr}_best_model.pth')
            print(f"Epoch: {epoch} Loss: {total_loss/len(dataloader)} Barlow Twins: {bt_loss/len(dataloader)} Contrastive Loss:{contrastive_loss/len(dataloader)}")
            torch.save(encoder.state_dict(),
                       f'{save_dir}/{args.encoder}_{args.batch_size}_{args.lr}_epoch_{epoch}.pth')
            scheduler.step()
            
        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=256)
    parser.add_argument("--lr",type=float,default=0.2)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--weight_decay",type=float,default=1e-5)
    parser.add_argument("--lam", type=float,default=5e-3)
    parser.add_argument("--encoder",type=str,default="resnet50")
    parser.add_argument("--seed", type=int,default=0)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--margin", type=float, default=0.0)
    args = parser.parse_args()
    print(args)
    main(args)
