import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import kagglehub
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

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


data_df = pd.get_dummies(data_df, columns=['BrewMethod', 'SugarScale'], drop_first=True)

Y_data = data_df['ABV']
X_data = data_df.drop('ABV', axis=1)

X = np.array(X_data.values.astype(np.float32))
Y = np.array(Y_data.values.astype(np.float32))


X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=42
)


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class BeerDataset(Dataset):
    def __init__(self, X, Y):

        self.X = torch.tensor(X, dtype=torch.float32).to(device)

        self.Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]



train_dataset = BeerDataset(X_train_scaled, y_train)
test_dataset = BeerDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


HIDDEN_NEURONS = 10


class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()

        self.input_layer = nn.Linear(input_size, HIDDEN_NEURONS)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(HIDDEN_NEURONS, 1)

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.linear(x)
        return x


model = MyModel(X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)


total_loss_train_plot = []
total_loss_test_plot = []
epochs = 10
print("Rozpoczynanie treningu...")
for epoch in range(epochs):
    total_loss_train = 0
    model.train()

    for inputs, labels in train_loader:

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

            predictions_val = model(inputs_val)
            batch_loss_val = criterion(predictions_val, labels_val)
            total_loss_val += batch_loss_val.item()

    avg_loss_val = total_loss_val / len(test_loader)
    total_loss_test_plot.append(avg_loss_val)

    print(f"Epoka {epoch + 1}/{epochs} | Trening MSE: {avg_loss_train:.4f} | Walidacja MSE: {avg_loss_val:.4f}")


plt.figure(figsize=(10, 6))
plt.plot(total_loss_train_plot, label='Train Loss (MSE)')
plt.plot(total_loss_test_plot, label='Test Loss (MSE)')
plt.title('Zbieżność Modelu: Mean Squared Error (MSE)')
plt.xlabel('Epoka')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()