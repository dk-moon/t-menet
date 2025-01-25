import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Data Load
print(os.getcwd())
data_path = "./t_menet/data/t_distribution_data.csv"
data = pd.read_csv(data_path)

# Data Preprocessing
le = LabelEncoder()
data["id"] = le.fit_transform(data["id"])
clusters = sorted(list(set(data["id"].values)))  # unique cluster id


# Torch Dataset Class
class SIMDataset(Dataset):
    def __init__(self, data, targets, clusters):
        self.data = data
        self.targets = targets
        self.clusters = clusters

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index], self.clusters[index]


x_col = ["id", "x"]  # input columns
y_col = ["y"]  # target columns

# Train-Test Split
SEED = 42
train, test = train_test_split(
    data, test_size=0.333, random_state=SEED, stratify=data["id"]
)

# Dataset Creation
train_dataset = SIMDataset(train[x_col].values, train[y_col].values, train["id"].values)
test_dataset = SIMDataset(test[x_col].values, test[y_col].values, test["id"].values)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define PyTorch MeNet Model
class MeNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=[20, 5]):
        super(MeNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# t-분포 손실 함수 정의
def t_loss(k):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        squared_error = torch.square(error)
        scaled_error = torch.log(k + squared_error)
        return torch.mean(scaled_error)

    return loss


# Model Training
def train_menet(
    model, train_loader, n_clusters, device, k=1.0, epochs=100, lr=0.001, patience=10
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = t_loss(k)  # Apply t-distribution loss

    # Initialize Random Effects
    output_dim = model.fc2.weight.data.size(0)  # Output dimension from fc2 weights
    b_hat = torch.zeros(n_clusters, output_dim, device=device)
    D_hat = torch.eye(output_dim, device=device)
    sig2e_est = 1.0

    best_loss = float("inf")
    wait = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            X_batch, y_batch, cluster_ids = batch
            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).float()

            # Forward pass to get Z (feature map)
            with torch.no_grad():
                feature_map = model.fc2(
                    model.relu(model.fc1(X_batch))
                )  # Shape: [batch_size, hidden_dim[1]]

            # Debugging Z shape
            # print(f"Feature Map shape: {feature_map.shape}, Expected shape: [batch_size, {output_dim}]")

            # E-Step: Update Random Effects
            for cluster_id in range(n_clusters):
                indices = (cluster_ids == cluster_id).nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    continue

                Z_i = feature_map[indices]  # Feature map for the current cluster
                y_i = y_batch[indices].squeeze(
                    -1
                )  # Remove extra dimension, Shape: [len(indices)]
                f_hat_i = model(X_batch[indices]).squeeze()  # Shape: [len(indices)]

                # Debugging shapes
                # print(f"Z_i shape: {Z_i.shape}, D_hat shape: {D_hat.shape}, y_i shape: {y_i.shape}, f_hat_i shape: {f_hat_i.shape}")

                # Ensure dimensions match for matrix multiplication
                V_hat_i = Z_i @ D_hat @ Z_i.T + sig2e_est * torch.eye(
                    Z_i.size(0), device=device
                )
                V_hat_inv_i = torch.linalg.inv(V_hat_i)

                # Expand (y_i - f_hat_i) to 2D for matrix multiplication
                residual = (y_i - f_hat_i).unsqueeze(-1)  # Shape: [len(indices), 1]

                # Update b_hat
                b_hat[cluster_id] = (D_hat @ Z_i.T @ V_hat_inv_i @ residual).squeeze(-1)

            # M-Step: Update Fixed Effects
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} : Loss {epoch_loss:.4f}")

        # Early Stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break

    return model, b_hat, sig2e_est


# Model Inference
def menet_predict(model, test_loader, b_hat, n_clusters, device):
    """
    Perform inference using the trained MeNet model with random effects.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test dataset.
        b_hat: Tensor containing random effect estimates for each cluster.
        n_clusters: Number of clusters (unique IDs).
        device: Device for computation (CPU or GPU).

    Returns:
        predictions: NumPy array of predictions for the test dataset.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            X_batch, _, cluster_ids = batch
            X_batch = X_batch.to(device).float()

            # Forward pass through the model to compute fixed effects
            y_pred = model(X_batch).squeeze()  # Shape: [batch_size]

            # Apply random effects to each cluster
            for cluster_id in range(n_clusters):
                indices = (cluster_ids == cluster_id).nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    continue

                # Apply random effects adjustment for the current cluster
                feature_map = model.fc2(
                    model.relu(model.fc1(X_batch))
                )  # Shape: [batch_size, hidden_dim[1]]
                Z_i = feature_map[indices]  # Feature map for the current cluster
                adjustment = Z_i @ b_hat[cluster_id].unsqueeze(
                    -1
                )  # Shape: [len(indices), 1]
                y_pred[indices] += adjustment.squeeze(-1)

            predictions.extend(y_pred.cpu().numpy())  # Convert predictions to NumPy

    return np.array(predictions)


# Main Execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(x_col)
n_clusters = len(clusters)
k = 11

# Model Initialization
model = MeNet(input_dim).to(device)

# Train the Model
model, b_hat, sig2e_est = train_menet(
    model, train_loader, n_clusters, device, k=k, epochs=100
)

# Predict
predictions = menet_predict(
    model=model,
    test_loader=test_loader,
    b_hat=b_hat,
    n_clusters=n_clusters,
    device=device,
)


# t-분포 손실 함수 정의
def t_loss_metric(y_true, y_pred, k=1.0):
    """
    Compute the t-distribution loss as a metric for evaluation.

    Args:
        y_true (np.array): Ground truth values.
        y_pred (np.array): Predicted values.
        k (float): Scaling parameter (degrees of freedom).

    Returns:
        float: t-distribution loss value.
    """
    error = y_true - y_pred
    squared_error = np.square(error)
    scaled_error = np.log(k + squared_error)
    return np.mean(scaled_error)


# Calculate evaluation metrics
def evaluate_predictions(y_true, y_pred, k=1.0):
    """
    Evaluate predictions using MSE, MAE, MAPE, MRPE, and t_loss.

    Args:
        y_true (np.array): Ground truth values.
        y_pred (np.array): Predicted values.
        k (float): Scaling parameter for t-distribution loss.

    Returns:
        metrics (dict): Dictionary of evaluation metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Avoid dividing by zero
    mrpe = np.mean(np.abs((y_true - y_pred) / y_pred)) * 100
    t_loss_value = t_loss_metric(y_true, y_pred, k)

    return {"MSE": mse, "MAE": mae, "MAPE": mape, "MRPE": mrpe, "t_loss": t_loss_value}


# Ground truth (y_test) should be prepared from your test dataset
y_test = test[y_col].values.flatten()

# Evaluate predictions
metrics = evaluate_predictions(y_test, predictions, k=k)

# Print metrics
print("===== Evaluation Metrics =====")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
