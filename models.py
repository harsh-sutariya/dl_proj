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


class Predictor(torch.nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.bl = nn.Bilinear(repr_dim,2,repr_dim*4)
        self.relu = nn.ReLU()
        # self.block1 = nn.Sequential(nn.Linear(repr_dim*4,repr_dim*2), nn.ReLU(),nn.Linear(repr_dim*2, repr_dim*4))
        # self.block2 = nn.Sequential(nn.Linear(repr_dim*4,repr_dim*2), nn.ReLU(),nn.Linear(repr_dim*2, repr_dim*4))
        self.linear = nn.Linear(repr_dim*4,repr_dim)
    def forward(self, embed, action):
        new_embed = self.bl(embed,action)
        new_embed = self.relu(new_embed)
        # new_embed = self.block1(new_embed)
        # new_embed = self.block2(new_embed) + new_embed
        next_embed = self.linear(new_embed)
        return next_embed

class Predictor_cross(torch.nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.action_embed = nn.Sequential(nn.Linear(2,repr_dim//2), nn.ReLU(),nn.Linear(repr_dim//2,repr_dim))
        self.mhsa = torch.nn.MultiheadAttention(repr_dim,num_heads=4,kdim=repr_dim,vdim=repr_dim, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(repr_dim, repr_dim*2), nn.ReLU(), torch.nn.Linear(repr_dim*2, repr_dim))
        self.norm1 = nn.LayerNorm(repr_dim)
        self.norm2 = nn.LayerNorm(repr_dim)

    def forward(self, embed, action):
        action_embed = self.action_embed(action) # B, 1024
        mhsa, _ = self.mhsa(action_embed, embed, embed) # B, 1024
        norm_mhsa = self.norm1(mhsa+action_embed) # B, 1024
        next_embed = self.norm2(self.linear(norm_mhsa)+norm_mhsa)
        return next_embed
class ViTEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = models.VisionTransformer(image_size=65,patch_size=5,num_layers=4,num_heads=8, hidden_dim=768, mlp_dim=3072)
        self.vit.conv_proj = nn.Conv2d(2,768,kernel_size=(5,5), stride=(5,5))
        
    def forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

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
        elif encoder=="ViT":
            self.enc = ViTEncoder()
            self.enc_dim = self.enc.vit.hidden_dim
        else:
            raise NotImplementedError
        self.repr_dim = 256
        self.fc = nn.Linear(self.enc_dim,self.repr_dim)
    def forward(self, img):
        return self.fc(self.enc(img))
    

class Baseline(torch.nn.Module):
    def __init__(self, device = "cuda", encoder="resnet18"):
        super().__init__()
        self.device = device
        if encoder=="resnet18":
            m = models.resnet18(weights="DEFAULT")
            self.encoder = nn.Sequential(*(list(m.children())[:-1]))
            self.repr_dim = 512
        utils.freeze_param(self.encoder)
        self.predictor = Predictor(self.repr_dim)
    
    def forward(self, states, actions, train=False):
        # print(states.shape)
        states = utils.preprocess_state(states)
        # print(states.shape)
        B, T, C, H, W = states.shape
        embeddings = self.encoder(states.reshape(-1,C, H, W)).reshape(B,T,-1)
        predicted_embeddings = [embeddings[:,0].unsqueeze(1)]
        if train:
            predictions = self.predictor(embeddings[:,:-1].reshape(-1,embeddings.shape[-1]), actions.reshape(-1,2)).reshape(B,T-1,self.repr_dim)
            # for i in range(actions.shape[1]):
            #     pred = self.predictor(embeddings[:,i],actions[:,i])
            #     predicted_embeddings.append(pred.unsqueeze(1))
            return torch.concat([embeddings[:,0].unsqueeze(1),predictions],dim=1),embeddings
        else:
            input_embed = embeddings[:,0]
            for i in range(actions.shape[1]):
                pred = self.predictor(input_embed,actions[:,i])
                predicted_embeddings.append(pred.unsqueeze(1))
            return torch.concat(predicted_embeddings,dim=1),embeddings
                
class JEPA(nn.Module):
    def __init__(self, encoder="resnet50", device = "cuda", context_encoder_path=None, freeze_encoder=False):
        super().__init__()
        self.device = device
        self.context_encoder = Encoder(encoder)
        if context_encoder_path:
            self.context_encoder.load_state_dict(torch.load(context_encoder_path, weights_only=True, map_location=self.device))
            if freeze_encoder:
                utils.freeze_param(self.context_encoder)
        self.context_encoder.to(device)
        self.repr_dim = self.context_encoder.repr_dim
        # self.predictor = Predictor_cross(self.repr_dim).to(device)
        self.predictor = Predictor(self.repr_dim).to(device)

    def forward(self, states, actions, train=False):
        # states = utils.preprocess_state(states)
        B, T, C, H, W = states.shape
        if train:
            embeddings = self.context_encoder(states.reshape(-1, C, H, W)).reshape(B,T,-1) #[s0,....sn]
            predictions = self.predictor(embeddings[:,:-1].reshape(-1,embeddings.shape[-1]), actions.reshape(-1,2)).reshape(B,T-1,self.repr_dim)
            return torch.concat([embeddings[:,0].unsqueeze(1),predictions],dim=1)
        s0 = self.context_encoder(states[:,0]) 
        predicted_embeddings = [s0.unsqueeze(1)]
        embed = s0
        for i in range(actions.shape[1]):
            embed = self.predictor(embed,actions[:,i])
            predicted_embeddings.append(embed.unsqueeze(1))
        pred = torch.concat(predicted_embeddings,dim=1)
        return pred


class JEPA_RNNCell(torch.nn.Module):
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
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(self.repr_dim*4, self.repr_dim))
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

class JEPA_RNNCell_seq(torch.nn.Module):
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
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(self.repr_dim*4, self.repr_dim))
        self.set_device()

    def forward(self, states, actions):
        B, T, _ = actions.shape
        s0 = self.context_encoder(states[:,0]) # B, Repr_dim
        # action_encoding = self.action_encoder(actions.reshape(-1,2)).reshape(B,T,-1)
        last_embed = s0
        predicted_embeddings = [s0.unsqueeze(1)]
        for t in range(T):
            input_embed = torch.concat([last_embed,self.action_encoder(actions[:,t])], dim=1)
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


class JEPA_RNN(torch.nn.Module):
    def __init__(self, encoder="resnet", rnn="gru",device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.context_encoder = Encoder(encoder)
        self.repr_dim = self.context_encoder.repr_dim
        self.action_encoder = nn.Sequential(nn.Linear(2,self.repr_dim), nn.ReLU(),nn.Linear(self.repr_dim,self.repr_dim//2))
        self.rnn = rnn
        if rnn == "rnn":
            self.predictor = nn.RNNCell(self.repr_dim//2,self.repr_dim)
        elif rnn == "lstm":
            self.predictor = nn.LSTMCell(self.repr_dim//2,self.repr_dim)
        elif rnn == "gru":
            self.predictor = nn.GRUCell(self.repr_dim//2,self.repr_dim)
        else:
            NotImplementedError
        # self.fc = nn.Sequential(nn.ReLU(), nn.Linear(self.repr_dim*4, self.repr_dim))
        self.set_device()

    def forward(self, states, actions):
        B, T, _ = actions.shape
        s0 = self.context_encoder(states[:,0]) # B, Repr_dim
        action_encoding = self.action_encoder(actions.reshape(-1,2)).reshape(B,T,-1)
        last_embed = s0
        predicted_embeddings = [s0.unsqueeze(1)]
        for t in range(T):
            if self.rnn == "lstm":
                last_embed, _ = self.predictor(action_encoding[:,t],(last_embed, torch.zeros_like(last_embed,device=self.device)))
            else:
                last_embed = self.predictor(action_encoding[:,t],last_embed)
            predicted_embeddings.append(last_embed.unsqueeze(1))
        predictions = torch.concat(predicted_embeddings, dim=1)
        return predictions
    
    def set_device(self):
        self.context_encoder.to(self.device)
        self.action_encoder.to(self.device)
        self.predictor.to(self.device)

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

