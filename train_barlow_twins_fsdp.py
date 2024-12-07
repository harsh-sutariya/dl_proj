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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import wandb
import time
import torchvision.transforms.v2 as v2



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(args, encoder, rank, world_size, train_loader, optimizer, epoch, transforms, sampler=None):
    encoder.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for idx, img in enumerate(dataloader):
        img1 = transforms(img)
        img2 = transforms(img)
        embed1 = encoder(img1)
        embed2 = encoder(img2)
        normalized_embed1 = (embed1 - embed1.mean(0))/embed1.std(0)
        normalized_embed2 = (embed2 - embed2.mean(0))/embed2.std(0)
        corr_matrix = torch.matmul(normalized_embed1.T,normalized_embed2)/embed1.shape[0]
        c_diff = (corr_matrix - torch.eye(encoder.repr_dim).to(rank)).pow(2)
        c_diff *= (off_diagonal*args.lam + torch.eye(encoder.repr_dim).to(rank))
        loss = c_diff.sum()
        ddp_loss[0] += loss
        ddp_loss[1] += len(img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        wandb.log({"loss":ddp_loss[0] / ddp_loss[1]})
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))
        
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    utils.seed_torch(args.seed, rank)
    transforms  = v2.RandomChoice([v2.ColorJitter(),v2.GaussianBlur(3),v2.RandomInvert(), v2.RandomAutocontrast()])
    data = dataset.ObsDataset(data_path="/scratch/DL24FA/train",device = rank)
    sampler = DistributedSampler(data, rank=rank, num_replicas=world_size, shuffle=True)
    dataloader = DataLoader(data,batch_size=args.batch_size,shuffle=False,pin_memory=True, num_workers=args.num_workers)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    
    encoder = models.Encoder(args.encoder).to(rank)
    encoder = FSDP(encoder)
    optimizer = optim.Adam(encoder.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    init_start_event.record()
    for epoch in range(args.epochs):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler)
    init_end_event.record()
    
def main(args):
    wandb.login()
    with wandb.init(project="encoder", entity="dl-nyu", config=vars(args)):
        start_time = time.time()
        utils.seed_numpy(args.seed)
        save_dir = os.path.join("../saved_models",str(start_time))
        os.mkdir(save_dir)
        
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
