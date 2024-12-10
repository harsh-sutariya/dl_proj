import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp.fully_sharded_data_parallel import (
#     CPUOffload,
#     BackwardPrefetch,
# )
# from torch.distributed.fsdp.wrap import (
#     size_based_auto_wrap_policy,
#     enable_wrap,
#     wrap,
# )
import wandb
import time
import torchvision.transforms.v2 as v2
import dataset
import utils
from torch.utils.data import DataLoader
import models
from tqdm import tqdm

t1 = v2.ColorJitter()
t2 = v2.GaussianBlur(3)
location_transform = v2.RandomAffine(90,translate=(0.25,0.25),interpolation=v2.InterpolationMode.BILINEAR)
cosine_similarity = nn.CosineSimilarity()
best_loss = float('inf')
save_dir = ""

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(args, encoder, rank, world_size, train_loader, optimizer, epoch, scheduler, sampler=None):
    encoder.train()
    ddp_loss = torch.zeros(4).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for idx, img in tqdm(enumerate(train_loader)):
        img1 = t1(img).to(rank)
        img2 = t2(img).to(rank)
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
        cos = torch.maximum(cosine_similarity(selected, embed3) + 1, torch.tensor([args.margin]*embed1.shape[0]))
        corr_matrix = torch.matmul(normalized_embed1.T,normalized_embed2)/embed1.shape[0]
        c_diff = (corr_matrix - torch.eye(embed1.shape[1]).to(rank)).pow(2)
        off_diagonal = (torch.ones((embed1.shape[1], embed1.shape[1]))-torch.eye(embed1.shape[1])).to(rank)
        c_diff *= (off_diagonal*args.lam + torch.eye(embed1.shape[1]).to(rank))
        bt_l = c_diff.sum()
        contr_l = cos.mean()
        loss = bt_l+ contr_l
        ddp_loss[2] += bt_l.item()
        ddp_loss[3]+= contr_l.item()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    dist.barrier()
    if rank == 0:
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        wandb.log({"loss":ddp_loss[0] / ddp_loss[1], "epoch":epoch, "lr": scheduler.get_last_lr()[0], "bt_loss": ddp_loss[2] / ddp_loss[1], "contrastive_loss":ddp_loss[3] / ddp_loss[1]})
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))
        if best_loss > ddp_loss[0] / ddp_loss[1]:
            best_loss = ddp_loss[0] / ddp_loss[1]
            torch.save(encoder.state_dict(),
                           f'{save_dir}/{args.encoder}_{args.batch_size}_{args.lr}_best_model.pth')
        torch.save(encoder.state_dict(),
                       f'{save_dir}/{args.encoder}_{args.batch_size}_{args.lr}_epoch_{epoch}.pth')
        
def ddp(rank, world_size, args):
    setup(rank, world_size)
    data = dataset.ObsDataset(data_path="/scratch/DL24FA/train")
    sampler = DistributedSampler(data, rank=rank, num_replicas=world_size, shuffle=True)
    dataloader = DataLoader(data,batch_size=args.batch_size,shuffle=False,pin_memory=False)
    # my_auto_wrap_policy = functools.partial(
    #     size_based_auto_wrap_policy, min_num_params=100
    # )
    torch.cuda.set_device(rank)


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    
    encoder = models.Encoder(args.encoder).to(rank)
    encoder = DDP(encoder, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    optimizer = optim.Adam(encoder.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,1e-5)
    init_start_event.record()
    for epoch in range(args.epochs):
        train(args, encoder, rank, world_size, dataloader, optimizer, epoch, scheduler, sampler=sampler)
        scheduler.step()
    init_end_event.record()
    cleanup()
    
def main(args):
    wandb.login()
    with wandb.init(project="encoder", entity="dl-nyu", config=vars(args)):
        start_time = time.time()
        utils.seed_numpy(args.seed)
        utils.seed_torch(args.seed)
        save_dir = os.path.join("../saved_models",str(start_time))
        os.mkdir(save_dir)
        WORLD_SIZE = torch.cuda.device_count()
        mp.spawn(ddp,
            args=(WORLD_SIZE, args),
            nprocs=WORLD_SIZE,
            join=True)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=256)
    parser.add_argument("--lr",type=float,default=0.2)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--weight_decay",type=float,default=1e-5)
    parser.add_argument("--lam", type=float,default=5e-3)
    parser.add_argument("--encoder",type=str,default="resnet50")
    parser.add_argument("--seed", type=int,default=0)
    args = parser.parse_args()
    print(args)
    main(args)
