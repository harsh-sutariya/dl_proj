from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import utils
import torch.nn as nn


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class ResNetEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = models.resnet18()
        self.enc.conv1 = nn.Conv2d(2,64, kernel_size=(7,7), padding=(3,3), stride=(2,2),bias=False)
        self.enc = nn.Sequential(*(list(self.enc.children())[:-1]))
    def forward(self, x):
        return self.enc(x).reshape(x.shape[0],-1)

class Encoder(nn.Module):
    def __init__(self, encoder="resnet", *kwargs):
        super().__init__()
        if encoder=="resnet":
            self.enc = ResNetEncoder()
            self.enc_dim = 512
        else:
            raise NotImplementedError
        self.repr_dim = 256
        self.fc = nn.Linear(self.enc_dim,self.repr_dim)
    def forward(self, img):
        return self.fc(self.enc(img))                  

class JEPA_RNNCell_tanh(torch.nn.Module):
    def __init__(self, encoder="resnet", rnn="gru",device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.context_encoder = Encoder(encoder)
        self.repr_dim = self.context_encoder.repr_dim
        self.action_encoder = nn.Sequential(nn.Linear(2,self.repr_dim*2), nn.ReLU(),nn.Linear(self.repr_dim*2,self.repr_dim))
        self.rnn = rnn
        if rnn == "rnn":
            self.predictor = nn.RNNCell(self.repr_dim*2,self.repr_dim*4)
        elif rnn == "lstm":
            self.predictor = nn.LSTMCell(self.repr_dim*2,self.repr_dim*4)
        elif rnn == "gru":
            self.predictor = nn.GRUCell(self.repr_dim*2,self.repr_dim*4)
        else:
            NotImplementedError
        self.fc = nn.Sequential(nn.Linear(self.repr_dim*4, self.repr_dim))
        self.set_device()

    def forward(self, states, actions):
        B, T, _ = actions.shape
        s0 = self.context_encoder(states[:,0]) # B, Repr_dim
        action_encoding = self.action_encoder(actions.reshape(-1,2)).reshape(B,T,-1)
        last_embed = s0
        predicted_embeddings = [s0.unsqueeze(1)]
        for t in range(T):
            input_embed = torch.concat([last_embed,action_encoding[:,t]], dim=1)
            if self.rnn == "lstm":
                latent_embed,_ = self.predictor(input_embed)
            else:
                latent_embed = self.predictor(input_embed)
            last_embed = self.fc(latent_embed)
            predicted_embeddings.append(last_embed.unsqueeze(1))
        predictions = torch.concat(predicted_embeddings, dim=1)
        return predictions
    
    def set_device(self):
        self.context_encoder.to(self.device)
        self.action_encoder.to(self.device)
        self.predictor.to(self.device)
        self.fc.to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

