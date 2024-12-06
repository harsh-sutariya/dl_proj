from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


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
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)

class ViTEncoder(nn.Module):
    def __init__(self, in_channels=2, img_size=64, patch_size=8, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, 
                                     dim_feedforward=1024, dropout=0.1),
            num_layers=6
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0]  # Return CLS token features

class RGC(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.hidden = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
    def forward(self, x, h):
        z = torch.cat([x, h], dim=-1)
        i = torch.sigmoid(self.input_gate(z))
        u = torch.sigmoid(self.update_gate(z))
        g = torch.tanh(self.hidden(z))
        h_new = u * h + (1-u) * (i * g)
        return h_new

class Predictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.rgc = RGC(hidden_dim, state_dim)
        self.out = nn.Linear(state_dim, state_dim)
        
    def forward(self, state, action):
        action_emb = self.action_proj(action)
        next_state = self.rgc(action_emb, state)
        return self.out(next_state)   

class JEPAModel(nn.Module):
    def __init__(self, device="cuda", repr_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        
        # Deeper ViT encoder
        self.encoder = ViTEncoder(
            in_channels=2,
            img_size=64,
            patch_size=4,  # Smaller patches
            embed_dim=repr_dim,
            num_layers=12,  # Increased from 6
            num_heads=8
        )
        self.target_encoder = ViTEncoder(embed_dim=repr_dim)
        
        # Copy parameters from encoder to target_encoder
        for param_q, param_k in zip(self.encoder.parameters(), 
                                  self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        self.predictor = Predictor(state_dim=repr_dim, action_dim=2)
        
    @torch.no_grad()
    def momentum_update(self, m=0.99):
        """Update target encoder using momentum update"""
        for param_q, param_k in zip(self.encoder.parameters(),
                                  self.target_encoder.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
            
    def forward(self, states, actions):
        """
        Args:
            states: [B, 1, Ch, H, W] - Initial state only
            actions: [B, T-1, 2] - Sequence of actions
        Returns:
            predictions: [B, T, D] - Predicted representations
        """
        B = states.shape[0]
        T = actions.shape[1] + 1
        
        # Get initial state representation
        init_state = self.encoder(states[:, 0])  # [B, D]
        
        # Prepare output tensor
        predictions = torch.zeros(B, T, self.repr_dim).to(self.device)
        predictions[:, 0] = init_state
        
        # Predict future states
        current_state = init_state
        for t in range(T-1):
            current_state = self.predictor(current_state, actions[:, t])
            predictions[:, t+1] = current_state
            
        return predictions


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