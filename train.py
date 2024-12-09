import torch
import torch.nn as nn
from tqdm import tqdm
from models import JEPA
from dataset import create_wall_dataloader
import wandb

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_data(device):
    data_path = "/scratch/DL24FA"
    training_set = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
    )
    return training_set

def distance(predictions, targets):
    
    # Invariance Loss (Alignment)
    invariance_loss = nn.MSELoss()(predictions, targets)

    # Variance Regularization
    mean_embedding = torch.mean(predictions, dim=0)
    variance_regularization = torch.mean((predictions - mean_embedding) ** 2)

    # Covariance Regularization
    z = predictions - predictions.mean(dim=0)
    covariance_matrix = (z.T @ z) / (z.size(0) - 1)
    off_diagonal_loss = torch.sum(covariance_matrix ** 2) - torch.sum(torch.diag(covariance_matrix) ** 2)

    vicreg_loss = invariance_loss + 0.1 * variance_regularization + 0.005 * off_diagonal_loss
    
    return vicreg_loss

def train_model(model, dataloader, optimizer, epochs, device):

    weights_path="JEPA_Trained.pth"

    running_loss = float("inf")
    model = model.to(device)

    for i in range(epochs):

        model.train()
        epoch_loss = 0.0

        for _, batch in tqdm(enumerate(dataloader)):

            states, locations, actions = batch
            states = states.to(device)
            locations = locations.to(device)
            actions = actions.to(device)
            
            predictions, targets = model(states, actions)
            loss = distance(predictions, targets)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            wandb.log({"Batch Loss": loss.item()})

        mean_loss = epoch_loss / len(dataloader)

        print(f"Epoch {i + 1}/{epochs}, Average Loss: {mean_loss:.7f}")
        wandb.log({"Per-Epoch Loss": mean_loss, "Epoch": i + 1})

        if mean_loss < running_loss:
            running_loss = mean_loss
            torch.save(model.state_dict(), weights_path)

def main():

    wandb.login()

    wandb.init(
        project="jepa",
        entity="dl-nyu",
        config={
            "epochs": 10,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "model": "JEPA",
        },
    )

    device = get_device()
    model = JEPA(inference = False)
    training_loader = load_data(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay= 1e-5)

    wandb.watch(model, log="all")

    train_model(model, training_loader, optimizer, epochs = 10, device = device)

    wandb.finish()

if __name__ == "__main__":
    main()
