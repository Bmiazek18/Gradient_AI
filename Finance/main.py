import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yfinance as yf
from copy import deepcopy as dc
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 1. Urządzenie
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# 2. Pobieranie danych
ticker_symbol = "NVDA"
data_raw = yf.download(ticker_symbol, period="10y")
data = data_raw[['Close']].copy()


# 3. Przygotowanie danych (Lagged Features)
def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df


lookback = 20
shifted_df = prepare_dataframe_for_lstm(data, lookback)
shifted_df_as_np = shifted_df.to_numpy()

# 4. Skalowanie
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(shifted_df_as_np)

# 5. Podział na X i Y
X = scaled_data[:, 1:]
Y = scaled_data[:, 0]

X = dc(np.flip(X, axis=1))


X = X.reshape((-1, lookback, 1))
Y = Y.reshape((-1, 1))

split_index = int(len(X) * 0.95)
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]


# 6. Dataset i DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y).float()

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.Y[i]


train_loader = DataLoader(TimeSeriesDataset(X_train, Y_train), batch_size=32, shuffle=True)


# 7. Model LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])



model = LSTM(1, 64, 2).to(device)

# 8. Trening
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

print(f"Trening na {device}...")
for epoch in range(50):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        outputs = model(x_batch)
        loss = loss_function(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Avg Loss: {running_loss / len(train_loader):.6f}')

# 9. Predykcja
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test).to(device).float()
    predictions = model(X_test_tensor).cpu().numpy()


dummy = np.zeros((len(predictions), lookback + 1))
dummy[:, 0] = predictions.flatten()
predictions_final = scaler.inverse_transform(dummy)[:, 0]

dummy_real = np.zeros((len(Y_test), lookback + 1))
dummy_real[:, 0] = Y_test.flatten()
y_test_final = scaler.inverse_transform(dummy_real)[:, 0]


test_dates = shifted_df.index[split_index:]

# 10. Wykres
plt.figure(figsize=(12, 6))


plt.plot(test_dates, y_test_final, label='Prawdziwa Cena', color='blue')
plt.plot(test_dates, predictions_final, label='Predykcja LSTM', color='red', linestyle='dashed')

plt.title('NVDA - Przewidywanie z poprawną osią czasu')
plt.xlabel('Data')
plt.ylabel('Cena ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()