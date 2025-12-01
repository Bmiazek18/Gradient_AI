import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import kagglehub
import os

HIDDEN_NEURONS_1 = 32
HIDDEN_NEURONS_2 = 16
LEARNING_RATE = 0.0005
EPOCHS = 50
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    dataset_dir = kagglehub.dataset_download("jtrofe/beer-recipes")
    csv_filename = 'recipeData.csv'
    full_path = os.path.join(dataset_dir, csv_filename)
    data_df = pd.read_csv(full_path, encoding="latin-1")
except Exception as e:
    print(f"Błąd ładowania z Kaggle: {e}. Upewnij się, że plik jest dostępny.")
    raise

data_df.dropna(inplace=True)

columns_to_drop = [
    'BeerID', 'Name', 'URL', 'Style', 'UserId', 'FG',
    'PrimingMethod', 'PrimingAmount'
]
data_df.drop(columns=columns_to_drop, inplace=True)

data_df = pd.get_dummies(data_df, columns=['BrewMethod', 'SugarScale','StyleID'], drop_first=True)

original_df = data_df.copy()

max_abv = original_df['ABV'].abs().max()

for col in data_df.columns:
    if data_df[col].dtype != 'bool':
        data_df[col] = data_df[col] / data_df[col].abs().max()


Y_data = data_df['ABV']
X_data = data_df.drop('ABV', axis=1)

X = np.array(X_data.values.astype(np.float32))
Y = np.array(Y_data.values.astype(np.float32))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
)

class BeerDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.Y = torch.tensor(Y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_dataset = BeerDataset(X_train, y_train)
test_dataset = BeerDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)



class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_NEURONS_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_NEURONS_1, HIDDEN_NEURONS_2)

        self.output_layer = nn.Linear(HIDDEN_NEURONS_2, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output_layer(x)
        return x


model = MyModel(X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

total_loss_train_plot = []
total_loss_test_plot = []

for epoch in range(EPOCHS):
    total_loss_train = 0
    model.train()

    for inputs, labels in train_loader:
        labels = labels.unsqueeze(1)
        optimizer.zero_grad()

        predictions = model(inputs)
        batch_loss = criterion(predictions, labels)

        total_loss_train += batch_loss.item()
        batch_loss.backward()
        optimizer.step()

    avg_loss_train = total_loss_train / len(train_loader)
    total_loss_train_plot.append(avg_loss_train)

    model.eval()
    total_loss_val = 0
    with torch.no_grad():
        for inputs_val, labels_val in test_loader:
            labels_val = labels_val.unsqueeze(1)
            predictions_val = model(inputs_val)
            batch_loss_val = criterion(predictions_val, labels_val)
            total_loss_val += batch_loss_val.item()

    avg_loss_val = total_loss_val / len(test_loader)
    total_loss_test_plot.append(avg_loss_val)

plt.figure(figsize=(10, 6))
plt.plot(total_loss_train_plot, label='Train Loss (MSE)')
plt.plot(total_loss_test_plot, label='Test Loss (MSE)')
plt.title('Zbieżność Modelu: Mean Squared Error (MSE) na przeskalowanych danych')
plt.xlabel('Epoka')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, labels in test_loader:
        labels = labels.unsqueeze(1)
        preds = model(inputs)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

all_preds = np.array(all_preds).flatten()
all_targets = np.array(all_targets).flatten()

all_preds_original = all_preds * max_abv
all_targets_original = all_targets * max_abv

plt.figure(figsize=(8, 6))
plt.scatter(all_targets_original, all_preds_original, alpha=0.6)
plt.xlabel("Rzeczywiste ABV (%)")
plt.ylabel("Przewidywane ABV (%)")
plt.title("Przewidywane vs Rzeczywiste ABV w skali procentowej")
plt.grid(True)

min_val = min(all_targets_original.min(), all_preds_original.min())
max_val = max(all_targets_original.max(), all_preds_original.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Idealna Predykcja')
plt.legend()
plt.show()