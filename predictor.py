import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from tabulate import tabulate

import warnings
warnings.filterwarnings("ignore")

# Load and split data
data = pd.read_csv(r'Data\prenorm.csv')
X_data = data.drop(['Overtakes', 'Country'], axis=1)
Y_data = data['Overtakes']

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, random_state=42, test_size=0.2) # Split 80/20

print("Data split")

# Normalize the data
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

print("Data normalized")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).unsqueeze(1)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).unsqueeze(1)

print("Data converted to tensors")


class OvertakePredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes, with_drop, with_norm, dropout_rate=0.2, ):
        super(OvertakePredictor, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if with_norm:
            layers.append(nn.LayerNorm(hidden_sizes[0]))  # Layer Normalization
        layers.append(nn.LeakyReLU()) # Activation
        if with_drop:
            layers.append(nn.Dropout(dropout_rate))  # Dropout after activation

        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            if with_norm:
                layers.append(nn.LayerNorm(hidden_sizes[i]))  # Layer Normalization
            layers.append(nn.LeakyReLU()) # Activation
            if with_drop:
                layers.append(nn.Dropout(dropout_rate))  # Dropout after activation

        layers.append(nn.Linear(hidden_sizes[-1], 1))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self): # He weight initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                init.constant_(layer.bias, 0.01)

    def forward(self, x):
        return self.network(x)

epochs = 1000

# Create PyTorch dataset
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

# Define DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Batches created\n")


def train_model(epochs):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_Y in train_loader:
            preds = model(batch_X)
            loss = criterion(preds, batch_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Sum batch losses

        scheduler.step()  # Update learning rate per epoch

        if epoch % 100 == 0:
            print(f"Epoch {epoch} of {epochs} - Loss: {epoch_loss / len(train_loader):.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")


def evaluate():
    model.eval()
    with torch.no_grad():
        preds_train = model(X_train_tensor).detach()
        preds_test = model(X_test_tensor).detach()

        train_mae = criterion(preds_train, Y_train_tensor).item()
        test_mae = criterion(preds_test, Y_test_tensor).item()

        return (round(train_mae, 4), round(test_mae, 4))

# Set to have layer normalization or dropout layer.

NORM = True 
DROP = True

if __name__ == '__main__':
    print(f"{X_data.head()}\n")
    print(f"{Y_data.head()}\n")

    print(f"Current parameters: LayerNorm - {NORM} | Dropout - {DROP}\n")

    hidden_combos = [[32], [64, 32], [128, 64, 32], [256, 128, 64, 32]]
    final_table = [("Hidden Layers", "Train MAE", "Test MAE")]

    for combo in hidden_combos:
        model = OvertakePredictor(X_train.shape[1], combo, NORM, DROP)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)
        criterion = torch.nn.L1Loss(reduction="mean")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        print(f"Hidden layer setup {combo}")

        train_model(epochs)
        print("\nTraining complete")
        a,b = evaluate()
        final_table.append((str(combo), a, b))
        print("Evaluation complete\n")
        print(f"Train MAE = {a}")
        print(f"Test MAE = {b}\n")

        print("-" * 50)
        print()

    table = tabulate(np.transpose(final_table), tablefmt= "rounded_grid")
    print(f"Scores for each layer setup with LayerNorm - {NORM} and Dropout - {DROP}\n")
    print(table)

    optimal = min(final_table[1:], key=lambda i: i[2])

    print(f"Optimal layers - {optimal[0]}")
    print(f"Resulting MAES - {optimal[1]:.4f}, {optimal[2]:.4f}")
