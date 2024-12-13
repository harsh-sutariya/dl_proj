import torch
import torch.nn as nn
from tqdm import tqdm
from models import JEPA
from dataset import create_wall_dataloader
import wandb
import argparse

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

def vicreg_distance(predictions, targets):
    
    # Invariance Loss (Alignment)
    invariance_loss = nn.MSELoss()(predictions, targets)

    # Variance Regularization
    mean_embedding = torch.mean(predictions, dim=0)
    variance_regularization = torch.mean((predictions - mean_embedding) ** 2)

    # Covariance Regularization
    z = predictions.reshape(-1, predictions.size(-1))
    z = z - z.mean(dim=0)
    covariance_matrix = (z.permute(1, 0) @ z) / (z.size(0) - 1)
    off_diagonal_loss = torch.sum(covariance_matrix ** 2) - torch.sum(torch.diag(covariance_matrix) ** 2)

    vicreg_loss = invariance_loss + 0.1 * variance_regularization + 0.005 * off_diagonal_loss
    
    return vicreg_loss

def barlow_twins_distance(predictions, targets, device):

    lam = 5e-3
    B, T, D = predictions.shape

    # Normalize Predictions & Targets
    normalized_predictions = (predictions - predictions.mean(dim=0)) / (predictions.std(dim=0) + 1e-6)
    normalized_targets = (targets - targets.mean(dim=0)) / (targets.std(dim=0) + 1e-6)

    # Cross-Correlation Matrix
    corr_matrix = torch.bmm(
        normalized_predictions.permute(1, 2, 0),
        normalized_targets.permute(1, 0, 2)
    ) / B

    # Off-Diagonal Penalty
    c_diff = (corr_matrix - torch.eye(D, device=device).reshape(1, D, D).repeat(T, 1, 1)).pow(2)
    off_diagonal = (torch.ones((D, D), device=device) - torch.eye(D, device=device)).reshape(1, D, D).repeat(T, 1, 1)
    c_diff *= (off_diagonal * lam + torch.eye(D, device=device).reshape(1, D, D).repeat(T, 1, 1))

    loss = c_diff.sum()
    return loss

def train_model(model, dataloader, optimizer, epochs, device, lr):

    # weights_path="JEPA_Trained.pth"
    weights_path = f"JEPA_Trained_{lr}.pth"

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
            # loss = vicreg_distance(predictions, targets)
            loss = barlow_twins_distance(predictions, targets, device)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update_target_encoder()
            
            wandb.log({"Batch Loss": loss.item()})

        mean_loss = epoch_loss / len(dataloader)

        print(f"Epoch {i + 1}/{epochs}, Average Loss: {mean_loss:.7f}")
        wandb.log({"Per-Epoch Loss": mean_loss, "Epoch": i + 1})

        if mean_loss < running_loss:
            running_loss = mean_loss
            torch.save(model.state_dict(), weights_path)

def main():

    parser = argparse.ArgumentParser(description="Train JEPA with configurable learning rate.")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for training.")
    args = parser.parse_args()

    lr = args.lr

    wandb.login()

    wandb.init(
        project="jepa",
        entity="dl-nyu",
        config={
            "epochs": 20,
            "learning_rate": lr,
            "weight_decay": 1e-5,
            "model": "JEPA",
        },
    )

    device = get_device()
    model = JEPA(inference = False)
    training_loader = load_data(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay= 1e-5)

    wandb.watch(model, log="all")

    train_model(model, training_loader, optimizer, epochs = 20, device = device, lr = lr)

    wandb.finish()

if __name__ == "__main__":
    main()
