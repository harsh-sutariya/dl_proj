import torchvision.transforms.v2 as v2
import torch
import os
import numpy as np

convert_rgb = v2.RGB()
def preprocess_img(img):
    """
    state: (2,H,W)
    """
    superimposed_img = torch.max(img,dim=0).values # (B,1,H,W)
    return convert_rgb(superimposed_img.unsqueeze(0))

def preprocess_state(state):
    """
    state: (B,T,2,H,W)
    """
    superimposed_img = torch.max(state,dim=2).values # (B,T,1,H,W)
    return convert_rgb(superimposed_img.unsqueeze(2))

def freeze_param(model):
    for param in model.parameters():
            param.requires_grad = False
def seed_torch(seed, device):
    """_summary_

    Args:
        seed (_type_): _description_
        device (_type_): _description_
    """
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_numpy(seed):
    """_summary_

    Args:
        seed (_type_): _description_
    """
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)