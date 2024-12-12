from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from timm import create_model

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


class Encoder(nn.Module):

    def __init__(self):

        super(Encoder, self).__init__()

        self.ViT = create_model("vit_large_patch16_224", pretrained=False, img_size=65, in_chans=2, num_classes=0)
        
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 768)
        )

    def forward(self, x):
        
        x = self.ViT(x)
        x = self.head(x)
        return x
                                                                     
class Predictor(nn.Module):
    
    def __init__(self):

        super(Predictor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(770, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )

    def forward(self, x):
        
        return self.network(x)

class JEPA(nn.Module):

    def __init__(self, inference):
        
        super(JEPA, self).__init__()

        self.encoder = Encoder()
        self.predictor = Predictor()
        self.target_encoder = Encoder()
        self.inference = inference
        self.repr_dim = 768

    def forward(self, states, actions):

        # States: [B, T, CH, H, W]
        # Actions: [B, T - 1, 2]

        B, T, _, _, _ = states.shape
        num_transitions = actions.shape[1]
        initial_state = states[:, 0]
        inference = self.inference
        
        if inference == False:
            
            # -------------------------------------------------------------------------------------------------------- Enc(theta) [Encoder]

            current_state_representation = self.encoder(initial_state)
            # Encoder Input: initial_state = states[:, 0] -> [B, CH, H, W] (First Time-Step)
            # Encoder Output: [B, D]

            # -------------------------------------------------------------------------------------------------------- Pred(phi) [Predictor]

            predicted_states = []

            for t in range(num_transitions):
                action_at_time_t = actions[:, t]
                predictor_input = torch.cat([current_state_representation, action_at_time_t], dim=-1)
                next_state_representation = self.predictor(predictor_input)
                predicted_states.append(next_state_representation)
                current_state_representation = next_state_representation

            predicted_states = torch.stack(predicted_states, dim=1)
            # Predictor Input: action_at_time_t = actions[:, t] -> [B, 2] & STATE_REP -> [B, D] || Concat: [B, D + 2]
            # Predictor Output: [B, D]
            # Predicted State List: T - 1 * [B, D]
            # Predicted States: [B, T - 1, D]
            
            # -------------------------------------------------------------------------------------------------------- Enc(psi) [Target Encoder]

            target_representation = []

            for t in range(T):
                state_at_time_t = states[:, t]
                target_representation.append(self.target_encoder(state_at_time_t))
            target_states = target_representation[1:]
            target_states = torch.stack(target_states, dim=1)
            
            # Target Encoder Input: states[:, t] [B, CH, H, W] (t Time-Step)
            # Target Encoder Output: [B, D]
            # Target Rep List: T * [B, D]
            # Target States: [B, T - 1, D]

            return predicted_states, target_states
        
        else:
            
            # -------------------------------------------------------------------------------------------------------- Enc(theta) [Trained Encoder]

            current_state_representation = self.encoder(initial_state)

            # -------------------------------------------------------------------------------------------------------- Pred(phi) [Trained Predictor]

            predicted_states = []

            for t in range(num_transitions):
                action_at_time_t = actions[:, t]
                predictor_input = torch.cat([current_state_representation, action_at_time_t], dim=-1)
                next_state_representation = self.predictor(predictor_input)
                predicted_states.append(next_state_representation)
                current_state_representation = next_state_representation
            
            predicted_states = torch.stack(predicted_states, dim=1)
            initial_state_representation = self.encoder(initial_state).unsqueeze(1)
            all_states = torch.cat([initial_state_representation, predicted_states], dim=1)
            
            return all_states
